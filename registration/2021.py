"""
Registration of Untracked 2D Laparoscopic Ultrasound to CT Images of the Liver
Using Multi-Labelled Content-Based Image Retrieval

本模块实现了 Ramalhinho et al. (2021) 提出的算法框架，包含以下核心组件：
  1. 特征编码：将血管截面编码为 (x, y, area) 三元组，并可附加类别标签（门静脉/肝静脉）
  2. 数据库生成：从 CT 血管模型中模拟超声平面，生成特征向量数据库
  3. 多标签 CBIR 检索：基于类别感知的距离度量进行图像检索
  4. HMM 优化：利用 Viterbi 算法在多帧图像序列中估计最优位姿序列

参考文献:
  Ramalhinho et al., "Registration of Untracked 2D Laparoscopic Ultrasound to CT
  Images of the Liver using Multi-Labelled Content-Based Image Retrieval,"
  IEEE TMI, 2021. DOI: 10.1109/TMI.2020.3045348
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import warnings


# =============================================================================
# 1. 数据结构定义
# =============================================================================

@dataclass
class VesselTriplet:
    """血管截面特征三元组: 2D 质心位置 + 面积"""
    x: float          # 质心 x 坐标 (mm)
    y: float          # 质心 y 坐标 (mm)
    area: float       # 截面面积 (mm^2)
    label: str = ""   # 血管类别标签，如 "portal", "hepatic"，空字符串表示无标签

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.area])


@dataclass
class FeatureVector:
    """
    特征向量：由多个 VesselTriplet 组成，支持多类别标签。
    对应论文中的 f 向量。
    """
    triplets: List[VesselTriplet] = field(default_factory=list)
    pose: Optional['ProbePose'] = None  # 关联的探头位姿（仅数据库条目有）

    def get_triplets_by_label(self) -> Dict[str, List[VesselTriplet]]:
        """按标签分组返回三元组"""
        groups = defaultdict(list)
        for t in self.triplets:
            groups[t.label].append(t)
        return dict(groups)

    def get_class_counts(self) -> Dict[str, int]:
        """返回每个类别的三元组数量 {M_c}"""
        return {label: len(ts) for label, ts in self.get_triplets_by_label().items()}

    def get_all_labels(self) -> List[str]:
        """获取所有出现的类别标签"""
        return list(self.get_triplets_by_label().keys())

    def get_centroid_array(self, label: str) -> np.ndarray:
        """获取指定标签的所有质心坐标数组，形状 (M_c, 2)"""
        ts = self.get_triplets_by_label().get(label, [])
        if not ts:
            return np.empty((0, 2))
        return np.array([[t.x, t.y] for t in ts])

    def get_area_array(self, label: str) -> np.ndarray:
        """获取指定标签的所有面积数组，形状 (M_c,)"""
        ts = self.get_triplets_by_label().get(label, [])
        if not ts:
            return np.empty(0)
        return np.array([t.area for t in ts])


@dataclass
class ProbePose:
    """
    探头位姿参数，对应论文中 (P_s, R, d) 的组合。
    """
    surface_point: np.ndarray   # 肝表面接触点 P (3D), shape (3,)
    rx: float                   # 绕 x 轴旋转角度 (度)
    ry: float                   # 绕 y 轴旋转角度 (度)
    rz: float                   # 绕 z 轴旋转角度 (度)
    depth: float                # 沿成像平面深度方向的平移 d (mm)

    @property
    def rotation(self) -> np.ndarray:
        """旋转角向量 [rx, ry, rz]"""
        return np.array([self.rx, self.ry, self.rz])

    @property
    def z_axis(self) -> np.ndarray:
        """计算成像平面的法向量 (z 轴方向)"""
        rx_rad = np.radians(self.rx)
        ry_rad = np.radians(self.ry)
        rz_rad = np.radians(self.rz)

        # 基础 z 轴 (成像平面法向量初始朝向)
        z = np.array([0.0, 0.0, 1.0])

        # 依次绕 x, y, z 轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx_rad), -np.sin(rx_rad)],
            [0, np.sin(rx_rad), np.cos(rx_rad)]
        ])
        Ry = np.array([
            [np.cos(ry_rad), 0, np.sin(ry_rad)],
            [0, 1, 0],
            [-np.sin(ry_rad), 0, np.cos(ry_rad)]
        ])
        Rz = np.array([
            [np.cos(rz_rad), -np.sin(rz_rad), 0],
            [np.sin(rz_rad), np.cos(rz_rad), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx @ z


@dataclass
class RetrievalResult:
    """单张 LUS 图像的检索结果"""
    input_feature: FeatureVector
    candidates: List[Tuple[ProbePose, float]]  # [(pose, distance), ...]
    K: int


# =============================================================================
# 2. 特征编码模块
# =============================================================================

class FeatureEncoder:
    """
    特征编码器：从分割的血管截面生成 FeatureVector。

    在实际应用中，血管截面通过图像分割获得。
    此处提供从分割结果构造特征向量的接口。
    """

    @staticmethod
    def from_segmentation(
        vessel_centroids: List[Tuple[float, float]],
        vessel_areas: List[float],
        vessel_labels: Optional[List[str]] = None
    ) -> FeatureVector:
        """
        从分割结果构造特征向量。

        参数:
            vessel_centroids: 血管截面质心列表 [(x, y), ...]，单位 mm
            vessel_areas: 血管截面面积列表 [a1, a2, ...]，单位 mm^2
            vessel_labels: 血管标签列表（可选），如 ["portal", "hepatic", ...]

        返回:
            FeatureVector 对象
        """
        assert len(vessel_centroids) == len(vessel_areas), \
            "质心和面积数量不匹配"
        if vessel_labels is not None:
            assert len(vessel_labels) == len(vessel_centroids), \
                "标签数量与质心数量不匹配"

        triplets = []
        for i, ((x, y), area) in enumerate(zip(vessel_centroids, vessel_areas)):
            label = vessel_labels[i] if vessel_labels is not None else ""
            triplets.append(VesselTriplet(x=x, y=y, area=area, label=label))

        return FeatureVector(triplets=triplets)


# =============================================================================
# 3. 数据库模拟模块
# =============================================================================

class DatabaseGenerator:
    """
    CBIR 数据库生成器。

    从 CT 血管模型中模拟 LUS 探头位姿，提取 2D 平面与血管模型的交截面，
    编码为特征向量并存储为可检索的数据库。

    数据库结构:
        - 无标签模式 (C=1): F = {F_M1}，按血管数量 M 分组
        - 有标签模式 (C=2): F = {F_Mh, F_Mp}，按各类别血管数量组合分组
    """

    def __init__(
        self,
        surface_points: np.ndarray,
        liver_surface_normals: np.ndarray,
        rx_range: Tuple[float, float] = (-40, 40),
        ry_range: Tuple[float, float] = (-90, 90),
        rz_range: Tuple[float, float] = (-40, 40),
        rx_step: float = 10.0,
        ry_step: float = 10.0,
        rz_step: float = 10.0,
        depth_range: Tuple[float, float] = (0, 30),
        depth_step: float = 5.0,
    ):
        """
        参数:
            surface_points: 肝表面采样点集合 P_s, shape (N_ps, 3)
            liver_surface_normals: 对应的法向量, shape (N_ps, 3)
            rx_range, ry_range, rz_range: 旋转角度范围 (度)
            rx_step, ry_step, rz_step: 旋转角度步长 (度)
            depth_range: 深度平移范围 (mm)
            depth_step: 深度平移步长 (mm)
        """
        self.surface_points = surface_points
        self.surface_normals = liver_surface_normals
        self.rx_range = rx_range
        self.ry_range = ry_range
        self.rz_range = rz_range
        self.rx_step = rx_step
        self.ry_step = ry_step
        self.rz_step = rz_step
        self.depth_range = depth_range
        self.depth_step = depth_step

        # 数据库: 按类别组合键 -> 特征向量列表
        # 键的格式: "M1_M2_..._MC" 表示各类别的血管数量
        self.database: Dict[str, List[FeatureVector]] = defaultdict(list)
        self.all_poses: List[ProbePose] = []
        self.all_features: List[FeatureVector] = []

    def _generate_pose_grid(self) -> List[ProbePose]:
        """生成所有位姿参数组合的网格"""
        poses = []
        rx_vals = np.arange(self.rx_range[0], self.rx_range[1] + self.rx_step / 2, self.rx_step)
        ry_vals = np.arange(self.ry_range[0], self.ry_range[1] + self.ry_step / 2, self.ry_step)
        rz_vals = np.arange(self.rz_range[0], self.rz_range[1] + self.rz_step / 2, self.rz_step)
        d_vals = np.arange(self.depth_range[0], self.depth_range[1] + self.depth_step / 2, self.depth_step)

        for i, ps in enumerate(self.surface_points):
            normal = self.surface_normals[i]
            for rx in rx_vals:
                for ry in ry_vals:
                    for rz in rz_vals:
                        for d in d_vals:
                            pose = ProbePose(
                                surface_point=ps.copy(),
                                rx=rx, ry=ry, rz=rz,
                                depth=d
                            )
                            poses.append(pose)
        return poses

    def generate(
        self,
        vessel_model_portal: Optional[np.ndarray] = None,
        vessel_model_hepatic: Optional[np.ndarray] = None,
        vessel_labels_model: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = (668, 544),
        pixel_size: float = 0.12,
    ) -> None:
        """
        生成完整的 CBIR 数据库。

        参数:
            vessel_model_portal: 门静脉 3D 模型点云, shape (N, 3) (可选)
            vessel_model_hepatic: 肝静脉 3D 模型点云, shape (N, 3) (可选)
            vessel_labels_model: 每个点的标签, shape (N,), 值为 "portal" 或 "hepatic"
                                 （若提供此项，则忽略 portal/hepatic 参数）
            image_size: 模拟 LUS 图像尺寸 (height, width)
            pixel_size: 像素大小 (mm/pixel)

        注意:
            此方法是框架的模拟接口。完整实现需要 3D 血管模型与平面的交计算，
           此处提供模拟数据生成的骨架结构。
        """
        poses = self._generate_pose_grid()
        print(f"[DatabaseGenerator] 共生成 {len(poses)} 个位姿参数组合")

        # 若提供了血管模型，使用 CT 重采样提取器
        use_ct_resampling = (
            (vessel_model_portal is not None and len(vessel_model_portal) > 0) or
            (vessel_model_hepatic is not None and len(vessel_model_hepatic) > 0)
        )

        if use_ct_resampling:
            extractor = CTResamplingFeatureExtractor(
                vessel_points_portal=vessel_model_portal if vessel_model_portal is not None else np.empty((0, 3)),
                vessel_points_hepatic=vessel_model_hepatic if vessel_model_hepatic is not None else np.empty((0, 3)),
                image_size=image_size,
                pixel_size=pixel_size,
            )
            print(f"[DatabaseGenerator] 使用 CT 重采样提取特征 "
                  f"(门静脉 {len(extractor.vessel_portal)} 点, "
                  f"肝静脉 {len(extractor.vessel_hepatic)} 点)")
        else:
            warnings.warn(
                "未提供 3D 血管模型，将使用合成随机数据演示。"
                "实际应用中请提供 CT 分割的血管点云模型。"
            )

        for pose in poses:
            if use_ct_resampling:
                fv = extractor.extract_feature(pose)
            else:
                fv = self._simulate_synthetic_feature(pose)

            if fv is not None and len(fv.triplets) > 0:
                fv.pose = pose
                self.all_poses.append(pose)
                self.all_features.append(fv)

                # 按类别数量组合分组存储
                key = self._make_db_key(fv)
                self.database[key].append(fv)

        total_entries = sum(len(v) for v in self.database.values())
        print(f"[DatabaseGenerator] 数据库构建完成: "
              f"{len(self.database)} 个分组, {total_entries} 条特征向量")

    def _simulate_synthetic_feature(
        self, pose: ProbePose, max_vessels: int = 8
    ) -> FeatureVector:
        """
        生成合成的特征向量用于演示。
        在实际应用中，此步骤应替换为从 CT 模型中提取真实血管截面。
        """
        np.random.seed(hash((tuple(pose.surface_point),
                             pose.rx, pose.ry, pose.rz, pose.depth)) % 2**31)

        n_portal = np.random.randint(1, max_vessels + 1)
        n_hepatic = np.random.randint(0, max_vessels + 1)

        triplets = []

        # 模拟门静脉截面
        for _ in range(n_portal):
            x = np.random.uniform(-80, 80)
            y = np.random.uniform(-80, 80)
            area = np.random.uniform(1.0, 30.0)
            triplets.append(VesselTriplet(x=x, y=y, area=area, label="portal"))

        # 模拟肝静脉截面
        for _ in range(n_hepatic):
            x = np.random.uniform(-80, 80)
            y = np.random.uniform(-80, 80)
            area = np.random.uniform(1.0, 30.0)
            triplets.append(VesselTriplet(x=x, y=y, area=area, label="hepatic"))

        return FeatureVector(triplets=triplets, pose=pose)

    def _extract_from_model(
        self,
        pose: ProbePose,
        vessel_model_portal: Optional[np.ndarray],
        vessel_model_hepatic: Optional[np.ndarray],
        vessel_labels_model: Optional[np.ndarray],
        image_size: Tuple[int, int],
        pixel_size: float,
    ) -> Optional[FeatureVector]:
        """
        从 3D 血管模型中提取 2D 平面的血管截面。

        流程:
        1. 根据位姿参数构造 2D 成像平面（原点、局部坐标系）
        2. 将 3D 血管点投影到平面，求到平面的距离
        3. 在可接受距离内对投影点聚类，识别各血管截面
        4. 计算每个截面的质心和面积
        5. 将 3D 质心投影到 2D 局部坐标
        """
        from scipy.spatial.distance import cdist
        from scipy.ndimage import label as ndlabel

        # 构造成像平面
        plane_origin, x_axis, y_axis, z_axis = CTResamplingFeatureExtractor.build_imaging_plane(pose)
        half_w = (image_size[1] * pixel_size) / 2.0
        half_h = (image_size[0] * pixel_size) / 2.0

        # 收集所有带标签的血管点
        all_points = []
        all_labels = []

        if vessel_labels_model is not None and len(vessel_labels_model) > 0:
            # 使用统一标签模型
            # 这里 vessel_labels_model 应为点云，通过参数区分
            pass

        if vessel_model_portal is not None and len(vessel_model_portal) > 0:
            all_points.append(vessel_model_portal)
            all_labels.extend(["portal"] * len(vessel_model_portal))

        if vessel_model_hepatic is not None and len(vessel_model_hepatic) > 0:
            all_points.append(vessel_model_hepatic)
            all_labels.extend(["hepatic"] * len(vessel_model_hepatic))

        if not all_points:
            return None

        points_3d = np.vstack(all_points)
        labels_arr = np.array(all_labels)

        # 将 3D 点投影到平面局部坐标
        rel = points_3d - plane_origin
        coords_local = np.column_stack([
            rel @ x_axis,
            rel @ y_axis,
            rel @ z_axis  # 沿深度方向
        ])

        # 仅保留在成像平面视野内的点
        in_fov = (
            (np.abs(coords_local[:, 0]) <= half_w) &
            (np.abs(coords_local[:, 1]) <= half_h) &
            (np.abs(coords_local[:, 2]) <= 15.0)  # 可接受的深度容差 (mm)
        )
        if not np.any(in_fov):
            return None

        fov_points = coords_local[in_fov]
        fov_labels = labels_arr[in_fov]

        # 对每个类别的点分别聚类，识别独立血管截面
        triplets = []
        unique_labels = np.unique(fov_labels)

        for lbl in unique_labels:
            mask = fov_labels == lbl
            pts_2d = fov_points[mask, :2]  # (N_c, 2) 在平面上的 x, y
            pts_z = fov_points[mask, 2]     # 沿深度的偏移

            if len(pts_2d) < 2:
                continue

            # 聚类：使用距离阈值将属于不同血管截面的点分开
            cluster_tol = 5.0  # mm，同一截面的点最大间距
            clusters = CTResamplingFeatureExtractor._cluster_points_2d(pts_2d, cluster_tol)

            for cluster_mask in clusters:
                cluster_pts = pts_2d[cluster_mask]
                cluster_z = pts_z[cluster_mask]

                if len(cluster_pts) < 2:
                    continue

                # 计算质心 (2D 平面坐标)
                centroid_2d = np.mean(cluster_pts, axis=0)

                # 计算面积：使用凸包或最小外接圆面积
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(cluster_pts)
                    area = hull.volume  # 2D 凸包面积
                except Exception:
                    # 退化为点集的散布半径面积
                    dists_from_center = np.linalg.norm(cluster_pts - centroid_2d, axis=1)
                    area = np.pi * np.max(dists_from_center) ** 2

                if area < 0.5:  # 过小的截面视为噪声
                    continue

                triplets.append(VesselTriplet(
                    x=centroid_2d[0],
                    y=centroid_2d[1],
                    area=area,
                    label=str(lbl)
                ))

        if not triplets:
            return None

        return FeatureVector(triplets=triplets)

    @staticmethod
    def _make_db_key(fv: FeatureVector) -> str:
        """
        构造数据库分组键。

        无标签: 返回 "M"（仅有血管总数）
        有标签: 返回 "label1:count1_label2:count2_..."（按标签字母序）
        """
        counts = fv.get_class_counts()
        if not counts:
            return "0"

        labels = sorted(counts.keys())
        if len(labels) == 1 and labels[0] == "":
            # 无标签模式
            return str(counts[""])
        else:
            # 有标签模式，包含标签名以避免解析歧义
            parts = []
            for l in sorted(counts.keys()):
                if l:  # 跳过空标签
                    parts.append(f"{l}:{counts[l]}")
            return "_".join(parts) if parts else "0"


# =============================================================================
# 3.5 CT 重采样特征提取模块
# =============================================================================

class CTResamplingFeatureExtractor:
    """
    CT 重采样特征提取器。

    从 CT 血管 3D 模型（点云）出发，根据探头位姿构造 2D 成像平面，
    提取平面与血管模型的交截面，生成多标签特征向量。

    这就是论文 Fig.1 上半部分描述的过程：
      对每个 (P_s, R, d) 组合 -> 生成虚拟超声切面 -> 提取血管截面特征 f

    输入:
      - CT 分割的 3D 血管点云（门静脉 / 肝静脉）
      - 探头位姿参数
    输出:
      - FeatureVector（每个候选位姿对应的特征向量，用于构建 CBIR 数据库）
    """

    def __init__(
        self,
        vessel_points_portal: np.ndarray,
        vessel_points_hepatic: np.ndarray,
        image_size: Tuple[int, int] = (668, 544),
        pixel_size: float = 0.12,
        cluster_tolerance: float = 5.0,
        depth_tolerance: float = 15.0,
        min_cluster_points: int = 3,
        min_section_area: float = 0.5,
    ):
        """
        参数:
            vessel_points_portal: 门静脉 3D 点云, shape (N_p, 3), 单位 mm
            vessel_points_hepatic: 肝静脉 3D 点云, shape (N_h, 3), 单位 mm
            image_size: 模拟 LUS 图像尺寸 (height, width), 单位像素
            pixel_size: 像素物理大小, 单位 mm/pixel
            cluster_tolerance: 聚类距离阈值 (mm)，同一血管截面内的点最大间距
            depth_tolerance: 深度容差 (mm)，点到成像平面的最大允许距离
            min_cluster_points: 最少点数，低于此数的聚类被忽略
            min_section_area: 最小截面面积 (mm^2)，低于此值的截面被忽略
        """
        self.vessel_portal = vessel_points_portal
        self.vessel_hepatic = vessel_points_hepatic
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.cluster_tolerance = cluster_tolerance
        self.depth_tolerance = depth_tolerance
        self.min_cluster_points = min_cluster_points
        self.min_section_area = min_section_area

        # 成像平面半宽半高 (mm)
        self.half_w = (image_size[1] * pixel_size) / 2.0
        self.half_h = (image_size[0] * pixel_size) / 2.0

    def extract_feature(self, pose: ProbePose) -> Optional[FeatureVector]:
        """
        对给定位姿，从 CT 血管模型中提取特征向量。

        完整流程:
        1. 根据位姿参数构造成像平面（原点 + 局部坐标系 x, y, z 轴）
        2. 将所有 3D 血管点投影到平面局部坐标
        3. 过滤：保留在视野内且深度偏差在容差内的点
        4. 按标签（门静脉/肝静脉）分组
        5. 对每组的 2D 投影点进行聚类，识别独立血管截面
        6. 对每个聚类计算质心和面积，生成 VesselTriplet

        参数:
            pose: 探头位姿

        返回:
            FeatureVector 或 None（若该位姿下无可见血管截面）
        """
        # 步骤 1: 构造成像平面
        plane_origin, x_axis, y_axis, z_axis = self.build_imaging_plane(pose)

        # 步骤 2-3: 投影并过滤所有血管点
        all_triplets = []

        # 处理门静脉
        if len(self.vessel_portal) > 0:
            triplets_p = self._process_vessel_class(
                self.vessel_portal, "portal", plane_origin, x_axis, y_axis, z_axis
            )
            all_triplets.extend(triplets_p)

        # 处理肝静脉
        if len(self.vessel_hepatic) > 0:
            triplets_h = self._process_vessel_class(
                self.vessel_hepatic, "hepatic", plane_origin, x_axis, y_axis, z_axis
            )
            all_triplets.extend(triplets_h)

        if not all_triplets:
            return None

        return FeatureVector(triplets=all_triplets)

    def _process_vessel_class(
        self,
        points_3d: np.ndarray,
        label: str,
        plane_origin: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
    ) -> List[VesselTriplet]:
        """
        处理单个类别的血管点：投影 -> 过滤 -> 聚类 -> 生成三元组。

        参数:
            points_3d: 该类别的 3D 血管点, shape (N, 3)
            label: 类别标签 ("portal" 或 "hepatic")
            plane_origin, x_axis, y_axis, z_axis: 成像平面参数

        返回:
            VesselTriplet 列表
        """
        # 投影到平面局部坐标
        rel = points_3d - plane_origin
        coords_local = np.column_stack([
            rel @ x_axis,   # 平面内 x
            rel @ y_axis,   # 平面内 y
            rel @ z_axis,   # 沿深度方向 z
        ])

        # 过滤：在视野内且深度偏差在容差内
        in_fov = (
            (np.abs(coords_local[:, 0]) <= self.half_w) &
            (np.abs(coords_local[:, 1]) <= self.half_h) &
            (np.abs(coords_local[:, 2]) <= self.depth_tolerance)
        )

        if np.sum(in_fov) < self.min_cluster_points:
            return []

        pts_2d = coords_local[in_fov, :2]

        # 聚类：识别独立血管截面
        clusters = self._cluster_points_2d(pts_2d, self.cluster_tolerance)

        triplets = []
        for cluster_mask in clusters:
            cluster_pts = pts_2d[cluster_mask]
            n_pts = np.sum(cluster_mask)

            if n_pts < self.min_cluster_points:
                continue

            # 计算质心
            centroid = np.mean(cluster_pts, axis=0)

            # 计算面积（凸包）
            area = self._compute_section_area(cluster_pts)

            if area < self.min_section_area:
                continue

            triplets.append(VesselTriplet(
                x=centroid[0],
                y=centroid[1],
                area=area,
                label=label
            ))

        return triplets

    @staticmethod
    def build_imaging_plane(pose: ProbePose) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        根据探头位姿构造成像平面。

        对应论文 Fig.1: 探头接触肝表面 P_s，沿法向量放置，
        然后施加旋转 R=[rx,ry,rz] 和深度平移 d。

        返回:
            (plane_origin, x_axis, y_axis, z_axis)
            - plane_origin: 平面原点 (3D), shape (3,)
            - x_axis: 平面内 x 轴单位向量
            - y_axis: 平面内 y 轴单位向量
            - z_axis: 平面法向量（深度方向）
        """
        # 构造旋转矩阵 R = Rz * Ry * Rx (ZYX 欧拉角)
        R = _rotation_matrix_zyx(pose.rx, pose.ry, pose.rz)

        # 初始坐标系：探头垂直于肝表面
        # 初始 z 轴 = 肝表面法向量（近似向上）
        z0 = np.array([0.0, 0.0, 1.0])
        x0 = np.array([1.0, 0.0, 0.0])
        y0 = np.array([0.0, 1.0, 0.0])

        # 施加旋转
        z_axis = R @ z0
        x_axis = R @ x0
        y_axis = R @ y0

        # 平面原点 = 表面接触点 + 深度偏移 * 法向量
        plane_origin = pose.surface_point + pose.depth * z_axis

        return plane_origin, x_axis, y_axis, z_axis

    @staticmethod
    def _cluster_points_2d(
        points: np.ndarray, tolerance: float
    ) -> List[np.ndarray]:
        """
        对 2D 点进行基于距离阈值的聚类。

        使用层次聚类思想：两两距离小于 tolerance 的点归为同一类。
        等价于构建距离图并提取连通分量。

        参数:
            points: 2D 点, shape (N, 2)
            tolerance: 距离阈值 (mm)

        返回:
            布尔掩码列表，每个掩码对应一个聚类
        """
        from scipy.spatial.distance import cdist

        n = len(points)
        if n == 0:
            return []

        # 计算距离矩阵
        dist_matrix = cdist(points, points)

        # 构建邻接关系
        adj = dist_matrix <= tolerance

        # BFS 提取连通分量
        visited = np.zeros(n, dtype=bool)
        clusters = []

        for start in range(n):
            if visited[start]:
                continue

            # BFS
            queue = [start]
            visited[start] = True
            component = [start]

            while queue:
                node = queue.pop(0)
                neighbors = np.where(adj[node] & ~visited)[0]
                for nb in neighbors:
                    visited[nb] = True
                    queue.append(nb)
                    component.append(nb)

            clusters.append(np.array(component))

        # 转换为布尔掩码列表
        masks = []
        for comp in clusters:
            mask = np.zeros(n, dtype=bool)
            mask[comp] = True
            masks.append(mask)

        return masks

    @staticmethod
    def _compute_section_area(points_2d: np.ndarray) -> float:
        """
        计算血管截面面积 (mm^2)。

        使用凸包面积作为截面面积的估计。
        若点数不足构成凸包，退化为最小外接圆面积。
        """
        from scipy.spatial import ConvexHull

        if len(points_2d) < 3:
            # 不足 3 点，用散布半径近似
            if len(points_2d) < 2:
                return 0.0
            centroid = np.mean(points_2d, axis=0)
            radius = np.max(np.linalg.norm(points_2d - centroid, axis=1))
            return np.pi * radius ** 2

        try:
            hull = ConvexHull(points_2d)
            return hull.volume  # 2D 中 volume 即面积
        except Exception:
            # 退化情况（共线点等）
            centroid = np.mean(points_2d, axis=0)
            radius = np.max(np.linalg.norm(points_2d - centroid, axis=1))
            return np.pi * radius ** 2


def _rotation_matrix_zyx(rx: float, ry: float, rz: float) -> np.ndarray:
    """构建 ZYX 欧拉角旋转矩阵 R = Rz @ Ry @ Rx"""
    rx_r, ry_r, rz_r = np.radians(rx), np.radians(ry), np.radians(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_r), -np.sin(rx_r)],
        [0, np.sin(rx_r), np.cos(rx_r)]
    ])
    Ry = np.array([
        [np.cos(ry_r), 0, np.sin(ry_r)],
        [0, 1, 0],
        [-np.sin(ry_r), 0, np.cos(ry_r)]
    ])
    Rz = np.array([
        [np.cos(rz_r), -np.sin(rz_r), 0],
        [np.sin(rz_r), np.cos(rz_r), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


# =============================================================================
# 4. 多标签 CBIR 检索模块 (论文 Eq. 1-5)
# =============================================================================

class MultiLabelledCBIR:
    """
    多标签内容_based 图像检索系统。

    实现论文 Section II-B 的距离度量 (Eq. 1-5):
      - Eq. 1: 确定大小向量 fS_c 和 fL_c
      - Eq. 2: 类别特定的 L2 距离 Delta
      - Eq. 3: 加权距离 D，包含面积惩罚项
      - Eq. 4-5: 检索策略，含搜索范围约束
    """

    def __init__(self, database: Dict[str, List[FeatureVector]], search_range: int = 2):
        """
        参数:
            database: CBIR 数据库 {key: [FeatureVector, ...]}
            search_range: 搜索范围 r，允许的各类别血管数量差异上限 (Eq. 5)
        """
        self.database = database
        self.search_range = search_range
    
    def distance(self, f1: FeatureVector, f2: FeatureVector) -> float:
        """
        # 对应论文中的公式4
        计算两个特征向量之间的多标签距离 D(f1, f2, C) (Eq. 3)。

        距离包含两部分:
          1. 各类别的 L2 匹配距离之和 (Eq. 2)
          2. 未匹配血管的面积惩罚项 (Eq. 4)

        参数:
            f1: 输入特征向量
            f2: 数据库中的特征向量

        返回:
            归一化距离值
        """
        # 获取所有类别标签
        labels1 = set(f1.get_class_counts().keys())
        labels2 = set(f2.get_class_counts().keys())
        all_labels = labels1 | labels2
        C = len(all_labels)

        if C == 0:
            return 0.0

        total_distance = 0.0
        total_area_L = 0.0
        total_area_S = 0.0

        for c in all_labels: #每个类别分别计算
            delta, a_l, a_s = self._class_distance(f1, f2, c)
            total_distance += delta
            total_area_L += a_l
            total_area_S += a_s

        # 面积惩罚项 (Eq. 4)
        if total_area_L > 0:
            area_penalty = total_area_L / total_area_S
        else:
            area_penalty = 1.0

        # 加权距离 D (Eq. 3)
        D = (total_distance / C) * area_penalty #swz, 这里和公式3不同，公式3中没有"/C"

        # 归一化 (Eq. 5 中的 min(M_I, M_T) 归一化)
        M1 = len(f1.triplets)
        M2 = len(f2.triplets)
        D_normalized = D / max(min(M1, M2), 1)

        return D_normalized

    @staticmethod
    def _class_distance(
        f1: FeatureVector, f2: FeatureVector, c: str
    ) -> Tuple[float, float, float]:
        """
        计算单个类别 c 的距离 Delta(f1, f2, c) (Eq. 2)。

        返回:
            (delta, A_L_c, A_S_c):
                delta - 类别距离
                A_L_c - 较大向量的总面积
                A_S_c - 匹配到的面积总和
        """
        # 获取类别 c 的三元组
        triplets1 = [t for t in f1.triplets if t.label == c]
        triplets2 = [t for t in f2.triplets if t.label == c]

        # Eq. 1: 根据同类别(c)的管腔数量，确定较小（fS_c）和较大向量（fL_c）
        if len(triplets1) <= len(triplets2):
            fS_c = triplets1
            fL_c = triplets2
        else:
            fS_c = triplets2
            fL_c = triplets1

        M_S_c = len(fS_c)   # 较小向量中类别c的管腔数
        M_L_c = len(fL_c)   # 较大向量中类别c的管腔数

        if M_S_c == 0:
            # 较小向量没有类别=c的管腔
            return 0.0, 0.0, 0.0

        # 获取 fL_c中各管腔的中心点，用于最近邻搜索
        if M_L_c > 0:
            L_centroids = np.array([[t.x, t.y] for t in fL_c])  # (M_L_c, 2)
        else:
            L_centroids = np.empty((0, 2))

        # Eq. 2: 计算 Delta
        delta_sum = 0.0
        A_S_matched = 0.0

        for i, t_s in enumerate(fS_c):
            centroid_s = np.array([t_s.x, t_s.y])

            if M_L_c > 0:
                # 计算 较大向量中每个管腔和较小向量中第i个管腔的中心点距离，确定距离最小的管腔在较大向量中的序号nearest_idx
                # 对应 公式2中的 m(.) 函数
                diffs = L_centroids - centroid_s  # (M_L_c, 2)
                dists = np.linalg.norm(diffs, axis=1)  # (M_L_c,)
                nearest_idx = np.argmin(dists)
                nearest_triplet = fL_c[nearest_idx]

                # 累加 L2 距离
                delta_sum += (t_s.area - nearest_triplet.area) ** 2 + dists[nearest_idx] ** 2
                A_S_matched += nearest_triplet.area  # 公式3中的分母
            else:
                delta_sum += t_s.area ** 2 + t_s.x ** 2 + t_s.y ** 2

        # Eq. 4: 计算面积项
        A_L_c = sum(t.area for t in fL_c) if M_L_c > 0 else 0.0

        return delta_sum, A_L_c, A_S_matched

    def retrieve(self, input_fv: FeatureVector, K: int = 1000) -> RetrievalResult:
        """
        从数据库中检索与输入特征向量最相似的 K 个候选位姿 (Eq. 5)。

        参数:
            input_fv: 输入 LUS 特征向量
            K: 返回的候选数量

        返回:
            RetrievalResult 对象
        """
        input_counts = input_fv.get_class_counts()
        all_labels = sorted(input_counts.keys())

        # 构建搜索目标子集 F_T (公式5)
        # 只搜索各类别数量差异不超过 search_range r 的列表
        candidate_distances = []

        # C数据库中的每幅管腔分割后的小图
        for db_key, fv_list in self.database.items():
            db_counts = self._parse_db_key(db_key)

            # 检查搜索范围约束 (Eq. 5)
            # 如果CT小图中和输入的LUS图像中，同类管腔数相差太大，丢弃此CT小图
            in_range = True
            for label in all_labels: #对每类管腔
                input_count = input_counts.get(label, 0)
                db_count = db_counts.get(label, 0)
                if abs(input_count - db_count) > self.search_range: 
                    in_range = False
                    break

            if not in_range:
                continue

            # 对子集中的每个特征向量计算距离
            for fv in fv_list:
                d = self.distance(input_fv, fv)  #公式5中的第1行
                if fv.pose is not None:
                    candidate_distances.append((fv.pose, d))

        # 按距离排序，取前 K 个
        candidate_distances.sort(key=lambda x: x[1])
        top_K = candidate_distances[:K]

        return RetrievalResult(
            input_feature=input_fv,
            candidates=top_K,
            K=len(top_K)
        )

    @staticmethod
    def _parse_db_key(key: str) -> Dict[str, int]:
        """解析数据库分组键为 {标签名: 血管数量} 字典"""
        if key == "0":
            return {}
        parts = key.split("_")
        counts = {}
        for p in parts:
            if ":" in p:
                label, count = p.rsplit(":", 1)
                try:
                    counts[label] = int(count)
                except ValueError:
                    pass
            else:
                # 无标签模式: 纯数字
                try:
                    counts[""] = int(p)
                except ValueError:
                    pass
        return counts

    def compute_precision(
        self,
        retrieval_result: RetrievalResult,
        ground_truth_pose: ProbePose,
        tre_threshold: float = 20.0,
        ct_landmarks: Optional[np.ndarray] = None,
        lus_landmarks: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算检索精度 (Precision)。

        精度定义为：在 K 个检索结果中，TRE 低于阈值的位姿比例。

        参数:
            retrieval_result: 检索结果
            ground_truth_pose: 真实位姿
            tre_threshold: TRE 阈值 (mm)，默认 20mm
            ct_landmarks: CT 空间中的标记点 (可选，用于精确 TRE 计算)
            lus_landmarks: LUS 空间中的标记点 (可选)

        返回:
            精度值 [0, 1]
        """
        if not retrieval_result.candidates:
            return 0.0

        accurate_count = 0
        for pose, dist in retrieval_result.candidates:
            tre = self._compute_tre(
                pose, ground_truth_pose, ct_landmarks, lus_landmarks
            )
            if tre <= tre_threshold:
                accurate_count += 1

        return accurate_count / len(retrieval_result.candidates)

    @staticmethod
    def _compute_tre(
        estimated_pose: ProbePose,
        ground_truth_pose: ProbePose,
        ct_landmarks: Optional[np.ndarray] = None,
        lus_landmarks: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算目标配准误差 (TRE)。

        如果提供了标记点，则使用完整的点配准误差计算；
        否则，使用位姿参数之间的欧氏距离作为近似。
        """
        if ct_landmarks is not None and lus_landmarks is not None:
            # 完整 TRE: 将 LUS 标记点变换到 CT 空间后计算 RMS 误差
            # 此处使用简化的位姿差异作为近似
            # 实际实现应使用完整的刚体变换 + 点投影
            pass

        # 简化近似: 使用接触点距离 + 角度差异
        pos_diff = np.linalg.norm(
            estimated_pose.surface_point - ground_truth_pose.surface_point
        )
        rot_diff = np.linalg.norm(estimated_pose.rotation - ground_truth_pose.rotation)

        # 加权组合作为 TRE 近似
        return pos_diff + rot_diff * 0.5


# =============================================================================
# 5. HMM 优化模块 (论文 Eq. 6-8)
# =============================================================================

class HMMPoseEstimator:
    """
    基于隐马尔可夫模型 (HMM) 的多帧位姿估计器。

    使用 Viterbi 算法求解 MAP 估计 (Eq. 6):
        argmin_{Jk1,...,JkN} sum(-log P(I_i | J_ki)) + sum(-log P(J_ki | J_ki-1))

    转移概率为多元高斯分布 (Eq. 7-8)，考虑:
      - 三维欧氏距离 (探头接触点)
      - 法向量角度差异
      - 前向运动约束
    """

    def __init__(
        self,
        sigma_z: float = 3.0,       # 成像平面法向 z 方向平移标准差 (mm)
        sigma_x: float = 0.6,       # 成像平面内 x 方向平移标准差 (mm) = 0.2 * sigma_z
        sigma_y: float = 0.6,       # 成像平面内 y 方向平移标准差 (mm) = 0.2 * sigma_z
        sigma_theta: float = 2.0,   # 角度标准差 (度)
    ):
        """
        参数:
            sigma_x: 成像平面内 x 方向平移标准差
            sigma_y: 成像平面内 y 方向平移标准差
            sigma_z: 沿成像平面法向量方向的平移标准差
            sigma_theta: 角度变化标准差
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.sigma_theta = sigma_theta

    def transition_cost(
        self,
        pose_prev: ProbePose,
        pose_curr: ProbePose,
        dt: float = 1.0,
        forward_direction: Optional[np.ndarray] = None,
    ) -> float:
        """
        对应公式7
        计算转移代价 -log P(J_ki | J_ki-1) (公式7-8)。

        参数:
            pose_prev: 前一帧的候选位姿
            pose_curr: 当前帧的候选位姿
            dt: 时间间隔 (秒)，用于缩放协方差
            forward_direction: 前向运动方向约束向量（可选）

        返回:
            转移代价 (非负浮点数)
        """
        # 公式8: 计算 delta 向量
        # 在前一帧位姿的旋转坐标系中计算位移
        delta = self._compute_delta(pose_prev, pose_curr, dt)

        # 前向运动约束
        if forward_direction is not None:
            movement_dir = pose_curr.surface_point - pose_prev.surface_point
            if np.linalg.norm(movement_dir) > 1e-6:
                movement_dir = movement_dir / np.linalg.norm(movement_dir)
                cos_angle = np.dot(movement_dir, forward_direction)
                if cos_angle < 0:  # 反向运动，角度 > 90 度
                    return np.inf

        # Eq. 7: 多元高斯转移概率的负对数
        # -log P = 0.5 * delta^T * Sigma^{-1} * delta + const
        # 其中 Sigma = |dt| * diag(sigma_x, sigma_y, sigma_z, sigma_theta)
        sigma_diag = np.array([self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_theta])
        cov = np.abs(dt) * np.diag(sigma_diag) #公式8的第2行
        cov_inv = np.diag(1.0 / sigma_diag) / np.abs(dt)

        cost = 0.5 * delta @ cov_inv @ delta

        return cost

    @staticmethod
    def _compute_delta(
        pose_prev: ProbePose, pose_curr: ProbePose, dt: float
    ) -> np.ndarray:
        """
        对应公式8
        计算状态差向量 delta (Eq. 8)。

        包含 4 个自由度:
          - 在前一帧旋转坐标系中投影的三维平移差
          - 两个平面法向量之间的角度差 theta

        返回:
            shape (4,) 的差向量 [dx, dy, dz, dtheta]
        """
        # 三维平移差
        translation_diff = pose_curr.surface_point - pose_prev.surface_point

        # 投影到前一帧的旋转坐标系
        # 构建前一帧的旋转矩阵
        R_prev = HMMPoseEstimator._rotation_matrix(
            pose_prev.rx, pose_prev.ry, pose_prev.rz
        )
        projected_diff = R_prev.T @ translation_diff  # 投影到前一帧局部坐标系

        # 角度差: 两个平面法向量的夹角
        z_prev = pose_prev.z_axis
        z_curr = pose_curr.z_axis

        cos_theta = np.clip(np.dot(z_prev, z_curr), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))

        return np.array([
            projected_diff[0],
            projected_diff[1],
            projected_diff[2],
            theta
        ])

    @staticmethod
    def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
        """构建 ZYX 欧拉角旋转矩阵"""
        rx_r, ry_r, rz_r = np.radians(rx), np.radians(ry), np.radians(rz)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx_r), -np.sin(rx_r)],
            [0, np.sin(rx_r), np.cos(rx_r)]
        ])
        Ry = np.array([
            [np.cos(ry_r), 0, np.sin(ry_r)],
            [0, 1, 0],
            [-np.sin(ry_r), 0, np.cos(ry_r)]
        ])
        Rz = np.array([
            [np.cos(rz_r), -np.sin(rz_r), 0],
            [np.sin(rz_r), np.cos(rz_r), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def viterbi(
        self,
        retrieval_results: List[RetrievalResult],
        timestamps: Optional[List[float]] = None,
    ) -> List[ProbePose]:
        """
        使用 Viterbi 算法估计最优位姿序列 (公式6)。

        参数:
            retrieval_results: N 帧图像的检索结果列表
            timestamps: 各帧时间戳列表 (可选)

        返回:
            最优位姿序列 [pose_1, pose_2, ..., pose_N]
        """
        N = len(retrieval_results)
        if N == 0:
            return []

        if timestamps is None:
            timestamps = [float(i) for i in range(N)]

        # 获取每个时间步的候选位姿
        candidates_per_step = []
        for rr in retrieval_results:
            poses = [pose for pose, dist in rr.candidates]
            candidates_per_step.append(poses)

        # 检查是否有空帧
        for t, cands in enumerate(candidates_per_step):
            if len(cands) == 0:
                warnings.warn(
                    f"第 {t} 帧无检索候选，跳过 HMM 优化。"
                    f"可尝试增大 search_range 或 K 值。"
                )
                return []

        # 计算前向运动方向（使用前两帧的最近邻估计）
        forward_direction = None
        if N >= 2:
            p1 = candidates_per_step[0][0]
            p2 = candidates_per_step[1][0]
            fwd = p2.surface_point - p1.surface_point
            if np.linalg.norm(fwd) > 1e-6:
                forward_direction = fwd / np.linalg.norm(fwd)

        INF = float('inf')

        # ---- 单次前向 DP，同时存储 backpointer ----
        # 节点代价 -log P(I_i | J_ki) = 0（论文中均匀先验）

        # t = 0: 初始化
        V_prev = np.zeros(len(candidates_per_step[0])) #迭代过程中，保存累加到上一帧的cost
        all_bps: List[np.ndarray] = []

        # t = 1, ..., N-1: 递推
        for t in range(1, N):
            Kt = len(candidates_per_step[t]) #当前帧的相似候帧数
            V_curr = np.full(Kt, INF)
            bp_curr = np.full(Kt, -1, dtype=int)  #当前帧每一个相似候选帧在上一帧的相似候选帧找到的最佳转移帧

            dt = timestamps[t] - timestamps[t - 1]
            fwd = forward_direction if t <= 2 else None# 论文II节最后一段
            #前向运动方向是根据序列前两帧的最近邻检索结果估计的，这是一个粗略的全局方向。
            # 只在前几帧强制约束可以防止明显反向的跳变，但不在后续帧持续施加，
            # 因为实际探头运动可能有小幅回退或转弯，强制全部帧都会过于严格。

            for k_curr in range(Kt):#依次处理当前帧的每一个相似候选帧 k_curr
                pose_curr = candidates_per_step[t][k_curr]

                for k_prev in range(len(candidates_per_step[t - 1])):  #依次处理上一帧的每一个候选相似帧k_prev
                    pose_prev = candidates_per_step[t - 1][k_prev]

                    #k_prev转移到k_curr的cost
                    trans_cost = self.transition_cost(
                        pose_prev, pose_curr, dt, fwd
                    )

                    total_cost = V_prev[k_prev] + trans_cost
                    if total_cost < V_curr[k_curr]:
                        V_curr[k_curr] = total_cost
                        bp_curr[k_curr] = k_prev

            all_bps.append(bp_curr)
            V_prev = V_curr

        # ---- 回溯最优路径 ----
        best_final = int(np.argmin(V_prev)) #最后一帧的最佳相似帧
        optimal_indices = [best_final]
        for t in range(N - 1, 0, -1):
            prev_idx = int(all_bps[t - 1][optimal_indices[0]])
            optimal_indices.insert(0, prev_idx)

        # 提取最优位姿
        optimal_poses = []
        for t in range(N):
            idx = optimal_indices[t]
            optimal_poses.append(candidates_per_step[t][idx])

        return optimal_poses

    def find_min_width_for_success(
        self,
        retrieval_results: List[RetrievalResult],
        ground_truth_pose: ProbePose,
        tre_threshold: float = 20.0,
        max_N: int = 10,
    ) -> Tuple[int, Optional[ProbePose], float]:
        """
        寻找成功配准所需的最小图像数量 N_s (论文 Section III-B 第二个实验)。

        逐步增加 HMM 宽度 N，直到第一帧的配准 TRE 低于阈值。

        返回:
            (N_s, estimated_pose, tre)
        """
        for n in range(1, min(max_N + 1, len(retrieval_results) + 1)):
            sub_results = retrieval_results[:n]
            poses = self.viterbi(sub_results)

            if poses:
                estimated_pose = poses[0]
                tre = self._compute_tre_simple(estimated_pose, ground_truth_pose)

                if tre <= tre_threshold:
                    return n, estimated_pose, tre

        # 未能成功配准
        if retrieval_results and retrieval_results[0].candidates:
            best_pose = retrieval_results[0].candidates[0][0]
            tre = self._compute_tre_simple(best_pose, ground_truth_pose)
            return max_N, best_pose, tre
        return max_N, None, float('inf')

    @staticmethod
    def _compute_tre_simple(pose_est: ProbePose, pose_gt: ProbePose) -> float:
        """简化的 TRE 计算（基于位姿参数差异）"""
        pos_diff = np.linalg.norm(pose_est.surface_point - pose_gt.surface_point)
        rot_diff = np.linalg.norm(pose_est.rotation - pose_gt.rotation)
        return pos_diff + 0.5 * rot_diff


# =============================================================================
# 6. 完整注册流程
# =============================================================================

class LUSCTRegistrationFramework:
    """
    LUS-CT 配准完整框架。

    整合数据库生成、CBIR 检索和 HMM 优化三个核心组件。
    """

    def __init__(
        self,
        search_range: int = 2,
        sigma_x: float = 0.6,
        sigma_y: float = 0.6,
        sigma_z: float = 3.0,
        sigma_theta: float = 2.0,
    ):
        self.search_range = search_range
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.sigma_theta = sigma_theta

        self.db_generator: Optional[DatabaseGenerator] = None
        self.cbir: Optional[MultiLabelledCBIR] = None
        self.hmm: Optional[HMMPoseEstimator] = None

    def build_database(
        self,
        surface_points: np.ndarray,
        surface_normals: np.ndarray,
        vessel_model_portal: Optional[np.ndarray] = None,
        vessel_model_hepatic: Optional[np.ndarray] = None,
        vessel_labels_model: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        步骤 1: 构建 CBIR 数据库 (术前)。

        参数:
            surface_points: 肝表面采样点, shape (N, 3)
            surface_normals: 法向量, shape (N, 3)
            vessel_model_portal: 门静脉 3D 点云 (可选)
            vessel_model_hepatic: 肝静脉 3D 点云 (可选)
            vessel_labels_model: 统一标签模型 (可选)
            **kwargs: 传递给 DatabaseGenerator 的额外参数 (rx_range 等)
        """
        self.db_generator = DatabaseGenerator(
            surface_points=surface_points,
            liver_surface_normals=surface_normals,
            **kwargs
        )
        self.db_generator.generate(
            vessel_model_portal=vessel_model_portal,
            vessel_model_hepatic=vessel_model_hepatic,
            vessel_labels_model=vessel_labels_model,
        )

        self.cbir = MultiLabelledCBIR(
            database=self.db_generator.database,
            search_range=self.search_range
        )
        self.hmm = HMMPoseEstimator(
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            sigma_z=self.sigma_z,
            sigma_theta=self.sigma_theta,
        )

    def register_single_image(
        self,
        lus_feature: FeatureVector,
        K: int = 200,
    ) -> Tuple[Optional[ProbePose], float]:
        """
        仅使用 CBIR 注册单张 LUS 图像 (无 HMM)。

        返回:
            (best_pose, best_distance)
        """
        if self.cbir is None:
            raise RuntimeError("数据库未构建，请先调用 build_database()")

        result = self.cbir.retrieve(lus_feature, K=K)
        if result.candidates:
            best_pose, best_dist = result.candidates[0]
            return best_pose, best_dist
        return None, float('inf')

    def register_sequence(
        self,
        lus_features: List[FeatureVector],
        timestamps: Optional[List[float]] = None,
        K: int = 200,
    ) -> List[ProbePose]:
        """
        步骤 2-3: 注册 LUS 图像序列 (术中)。

        对每张图像执行 CBIR 检索，然后使用 HMM 优化位姿序列。

        参数:
            lus_features: LUS 特征向量序列
            timestamps: 时间戳序列 (可选)
            K: 每张图像检索的候选数量

        返回:
            最优位姿序列
        """
        if self.cbir is None or self.hmm is None:
            raise RuntimeError("数据库未构建，请先调用 build_database()")

        # 对每张图像执行 CBIR 检索
        retrieval_results = []
        for fv in lus_features:
            result = self.cbir.retrieve(fv, K=K)
            retrieval_results.append(result)

        # 使用 HMM 优化位姿序列
        optimal_poses = self.hmm.viterbi(retrieval_results, timestamps)

        return optimal_poses


# =============================================================================
# 7. 演示与测试
# =============================================================================

def _generate_synthetic_vessel_tree(
    n_points: int = 2000,
    center: np.ndarray = None,
    radius_range: Tuple[float, float] = (10, 60),
    seed: int = 42,
) -> np.ndarray:
    """
    生成合成的 3D 血管树点云，用于演示。

    模拟一棵从中心向外辐射的树状血管结构。

    返回:
        points: shape (N, 3)
    """
    if center is None:
        center = np.array([0.0, 0.0, 50.0])

    rng = np.random.RandomState(seed)

    # 主干 + 分支结构
    points = []

    # 主干: 沿 z 轴方向的管道
    n_trunk = n_points // 4
    z_trunk = rng.uniform(center[2] - 40, center[2] + 30, n_trunk)
    x_trunk = center[0] + rng.randn(n_trunk) * 3
    y_trunk = center[1] + rng.randn(n_trunk) * 3
    points.append(np.column_stack([x_trunk, y_trunk, z_trunk]))

    # 分支 1: 沿 +x 方向
    n_br = n_points // 6
    t = rng.uniform(0, 1, n_br)
    points.append(np.column_stack([
        center[0] + t * 50 + rng.randn(n_br) * 2,
        center[1] + rng.randn(n_br) * 2,
        center[2] - t * 20 + rng.randn(n_br) * 2
    ]))

    # 分支 2: 沿 -x 方向
    points.append(np.column_stack([
        center[0] - t * 45 + rng.randn(n_br) * 2,
        center[1] + rng.randn(n_br) * 2,
        center[2] - t * 15 + rng.randn(n_br) * 2
    ]))

    # 分支 3: 沿 +y 方向
    t2 = rng.uniform(0, 1, n_br)
    points.append(np.column_stack([
        center[0] + rng.randn(n_br) * 2,
        center[1] + t2 * 40 + rng.randn(n_br) * 2,
        center[2] - t2 * 10 + rng.randn(n_br) * 2
    ]))

    # 分支 4: 沿 -y 方向
    points.append(np.column_stack([
        center[0] + rng.randn(n_br) * 2,
        center[1] - t2 * 35 + rng.randn(n_br) * 2,
        center[2] - t2 * 10 + rng.randn(n_br) * 2
    ]))

    # 剩余点: 小分支和噪声
    n_remain = n_points - len(np.vstack(points))
    if n_remain > 0:
        theta = rng.uniform(0, 2 * np.pi, n_remain)
        r = rng.uniform(*radius_range, n_remain)
        points.append(np.column_stack([
            center[0] + r * np.cos(theta) + rng.randn(n_remain) * 2,
            center[1] + r * np.sin(theta) + rng.randn(n_remain) * 2,
            center[2] + rng.randn(n_remain) * 15
        ]))

    return np.vstack(points)


def demo_synthetic():
    """
    使用合成 3D 血管模型演示完整的注册流程：

    完整流程对应论文 Fig.1:
      1. 生成合成肝表面 + 合成 3D 血管点云（模拟 CT 分割结果）
      2. CT 重采样：对每个候选位姿构造成像平面，从 3D 血管模型中提取
         血管截面特征，构建 CBIR 数据库（每个库条目对应一个虚拟超声切面）
      3. 构造 LUS 特征向量（模拟超声图像分割后的结果）
      4. CBIR 检索：在数据库中搜索与 LUS 特征最匹配的 K 个候选
      5. HMM 优化：利用多帧序列的运动先验，估计最优位姿
    """
    print("=" * 70)
    print("LUS-CT 配准框架演示 — 完整 CT 重采样流程")
    print("=" * 70)

    # ==================================================================
    # 步骤 1: 生成合成数据（模拟 CT 分割结果）
    # ==================================================================
    print("\n[1] 生成合成 CT 数据（肝表面 + 血管模型）...")

    np.random.seed(42)
    n_surface_points = 50

    # 合成肝表面: 近似半球面（右叶）
    theta = np.random.uniform(0, np.pi / 2, n_surface_points)
    phi = np.random.uniform(-np.pi / 4, np.pi / 4, n_surface_points)
    radius = 80.0

    surface_points = np.column_stack([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta)
    ])
    surface_normals = surface_points / np.linalg.norm(
        surface_points, axis=1, keepdims=True
    )
    print(f"  肝表面采样点: {n_surface_points} 个")

    # 合成门静脉 3D 点云（模拟 CT 分割后的门静脉模型）
    vessel_portal = _generate_synthetic_vessel_tree(
        n_points=3000, center=np.array([10.0, -5.0, 55.0]),
        radius_range=(10, 55), seed=42
    )
    print(f"  门静脉 3D 点云: {len(vessel_portal)} 个点")

    # 合成肝静脉 3D 点云（模拟 CT 分割后的肝静脉模型）
    vessel_hepatic = _generate_synthetic_vessel_tree(
        n_points=2000, center=np.array([-5.0, 10.0, 60.0]),
        radius_range=(8, 50), seed=123
    )
    print(f"  肝静脉 3D 点云: {len(vessel_hepatic)} 个点")

    # ==================================================================
    # 步骤 2: CT 重采样 — 构建候选数据库
    # ==================================================================
    print("\n[2] CT 重采样：构建候选特征数据库...")
    print("  (对每个位姿参数组合，从 CT 血管模型中提取虚拟超声切面特征)")

    framework = LUSCTRegistrationFramework(
        search_range=3,  # 合成演示使用稍大搜索范围
        sigma_x=0.6,
        sigma_y=0.6,
        sigma_z=3.0,
        sigma_theta=2.0,
    )

    # 使用 CT 重采样提取器构建数据库（而非合成随机数据）
    framework.build_database(
        surface_points=surface_points,
        surface_normals=surface_normals,
        vessel_model_portal=vessel_portal,
        vessel_model_hepatic=vessel_hepatic,
        rx_range=(-20, 20),
        ry_range=(-20, 20),
        rz_range=(-20, 20),
        rx_step=20,
        ry_step=20,
        rz_step=20,
        depth_range=(0, 10),
        depth_step=10,
    )

    # 展示数据库统计
    db = framework.db_generator
    print(f"\n  数据库统计:")
    print(f"    位姿组合总数: {len(db.all_poses)}")
    print(f"    有效特征向量: {len(db.all_features)}")
    if db.all_features:
        counts_p = [f.get_class_counts().get("portal", 0) for f in db.all_features]
        counts_h = [f.get_class_counts().get("hepatic", 0) for f in db.all_features]
        print(f"    每条特征平均门静脉截面: {np.mean(counts_p):.1f}")
        print(f"    每条特征平均肝静脉截面: {np.mean(counts_h):.1f}")

    # ==================================================================
    # 步骤 3: 模拟 LUS 输入特征向量
    # ==================================================================
    print(f"\n[3] 构造输入 LUS 特征向量（模拟超声分割结果）...")

    # 模拟 3 帧连续 LUS: 使用数据库中某条特征加噪声，模拟真实 LUS 分割
    n_frames = 3
    lus_features = []

    # 从数据库中选取一个"真实"位姿对应的特征，加噪声作为 LUS 输入
    # 这样可以验证检索是否能找到接近的候选
    if len(db.all_features) > 10:
        base_fv = db.all_features[5]  # 选取第 6 条作为"真实"
        base_pose = db.all_poses[5]

        for i in range(n_frames):
            rng = np.random.RandomState(200 + i)
            triplets = []
            for t in base_fv.triplets:
                # 添加位置噪声 (模拟 LUS 分割误差)
                triplets.append(VesselTriplet(
                    x=t.x + rng.randn() * 1.5,
                    y=t.y + rng.randn() * 1.5,
                    area=max(t.area + rng.randn() * 1.0, 0.5),
                    label=t.label
                ))
            fv = FeatureVector(triplets=triplets)
            lus_features.append(fv)

            counts = fv.get_class_counts()
            n_p = counts.get("portal", 0)
            n_h = counts.get("hepatic", 0)
            print(f"  帧 {i + 1}: {n_p} 门静脉 + {n_h} 肝静脉 = "
                  f"{len(triplets)} 个血管截面")
    else:
        print("  数据库条目过少，无法模拟 LUS 输入")
        return

    # ==================================================================
    # 步骤 4: CBIR 单帧检索
    # ==================================================================
    print(f"\n[4] CBIR 单帧检索 (K=5)...")

    for i, fv in enumerate(lus_features):
        pose, dist = framework.register_single_image(fv, K=5)
        if pose is not None:
            print(f"  帧 {i + 1}: 最近邻位姿 P=({pose.surface_point[0]:.1f}, "
                  f"{pose.surface_point[1]:.1f}, {pose.surface_point[2]:.1f}), "
                  f"R=({pose.rx:.0f}, {pose.ry:.0f}, {pose.rz:.0f}), "
                  f"d={pose.depth:.0f}, 距离={dist:.4f}")
        else:
            print(f"  帧 {i + 1}: 未找到匹配")

    # ==================================================================
    # 步骤 5: HMM 序列优化
    # ==================================================================
    print(f"\n[5] HMM 序列优化 (N={n_frames}, K=50)...")

    timestamps = [0.0, 0.5, 1.0]
    optimal_poses = framework.register_sequence(
        lus_features, timestamps=timestamps, K=50
    )

    if optimal_poses:
        for i, pose in enumerate(optimal_poses):
            print(f"  帧 {i + 1} 最优位姿: P=({pose.surface_point[0]:.1f}, "
                  f"{pose.surface_point[1]:.1f}, {pose.surface_point[2]:.1f}), "
                  f"R=({pose.rx:.0f}, {pose.ry:.0f}, {pose.rz:.0f}), "
                  f"d={pose.depth:.0f}")
    else:
        print("  HMM 未能找到有效路径（某些帧无匹配候选）")

    # ==================================================================
    # 步骤 6: 距离度量演示
    # ==================================================================
    print(f"\n[6] 多标签距离度量演示...")

    f1 = FeatureVector(triplets=[
        VesselTriplet(x=10.0, y=20.0, area=5.0, label="portal"),
        VesselTriplet(x=-15.0, y=5.0, area=8.0, label="portal"),
        VesselTriplet(x=30.0, y=-10.0, area=3.0, label="hepatic"),
    ])
    f2 = FeatureVector(triplets=[
        VesselTriplet(x=11.0, y=21.0, area=5.5, label="portal"),
        VesselTriplet(x=-14.0, y=6.0, area=7.5, label="portal"),
        VesselTriplet(x=31.0, y=-9.0, area=3.2, label="hepatic"),
        VesselTriplet(x=5.0, y=-20.0, area=4.0, label="hepatic"),
    ])

    d = framework.cbir.distance(f1, f2)
    print(f"  f1 (2P + 1H) vs f2 (2P + 2H): 距离 D = {d:.4f}")

    d_same = framework.cbir.distance(f1, f1)
    print(f"  f1 vs f1 (自身): 距离 D = {d_same:.4f}")

    f3 = FeatureVector(triplets=[
        VesselTriplet(x=-50.0, y=-50.0, area=1.0, label="portal"),
        VesselTriplet(x=50.0, y=50.0, area=30.0, label="hepatic"),
    ])
    d_diff = framework.cbir.distance(f1, f3)
    print(f"  f1 vs f3 (完全不同): 距离 D = {d_diff:.4f}")

    f1_unlabelled = FeatureVector(triplets=[
        VesselTriplet(x=t.x, y=t.y, area=t.area, label="")
        for t in f1.triplets
    ])
    f2_unlabelled = FeatureVector(triplets=[
        VesselTriplet(x=t.x, y=t.y, area=t.area, label="")
        for t in f2.triplets
    ])
    d_unlabelled = framework.cbir.distance(f1_unlabelled, f2_unlabelled)
    print(f"  f1 vs f2 (无标签): 距离 D = {d_unlabelled:.4f}")
    print(f"  → 标签模式({d:.4f}) vs 无标签模式({d_unlabelled:.4f}): "
          f"标签模式距离{'更小' if d < d_unlabelled else '更大'}")

    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)


def demo_hmm_viterbi():
    """
    演示 HMM Viterbi 算法的详细执行过程。
    """
    print("\n" + "=" * 70)
    print("HMM Viterbi 算法演示")
    print("=" * 70)

    hmm = HMMPoseEstimator(sigma_x=0.6, sigma_y=0.6, sigma_z=3.0, sigma_theta=2.0)

    # 构造 3 帧的候选位姿
    np.random.seed(42)

    # 创建一条"真实"运动轨迹
    true_poses = []
    base_point = np.array([40.0, 20.0, 60.0])
    for i in range(3):
        pose = ProbePose(
            surface_point=base_point + np.array([i * 5.0, i * 2.0, i * 1.0]),
            rx=0.0, ry=0.0, rz=0.0,
            depth=0.0
        )
        true_poses.append(pose)

    # 为每帧创建 K=5 个候选位姿（包含真实位姿 + 噪声位姿）
    K = 5
    retrieval_results = []

    for t in range(3):
        candidates = []
        for k in range(K):
            if k == t:  # 在第 t 帧的第 t 个位置放入"接近真实"的位姿
                pose = ProbePose(
                    surface_point=true_poses[t].surface_point + np.random.randn(3) * 2,
                    rx=np.random.randn() * 5,
                    ry=np.random.randn() * 5,
                    rz=np.random.randn() * 5,
                    depth=0.0
                )
            else:
                # 随机偏远的位姿
                pose = ProbePose(
                    surface_point=np.array([
                        np.random.uniform(-60, 60),
                        np.random.uniform(-60, 60),
                        np.random.uniform(40, 80)
                    ]),
                    rx=np.random.uniform(-40, 40),
                    ry=np.random.uniform(-40, 40),
                    rz=np.random.uniform(-40, 40),
                    depth=np.random.uniform(0, 10)
                )
            candidates.append((pose, np.random.rand()))

        retrieval_results.append(RetrievalResult(
            input_feature=FeatureVector(),
            candidates=candidates,
            K=K
        ))

    # 运行 Viterbi
    timestamps = [0.0, 0.5, 1.0]
    optimal = hmm.viterbi(retrieval_results, timestamps)

    print(f"\n输入: {len(retrieval_results)} 帧, 每帧 {K} 个候选")
    print("\n估计的最优位姿序列:")
    for i, pose in enumerate(optimal):
        print(f"  帧 {i + 1}: P=({pose.surface_point[0]:.1f}, "
              f"{pose.surface_point[1]:.1f}, {pose.surface_point[2]:.1f}), "
              f"R=({pose.rx:.1f}, {pose.ry:.1f}, {pose.rz:.1f})")

    print("\n真实位姿序列:")
    for i, pose in enumerate(true_poses):
        print(f"  帧 {i + 1}: P=({pose.surface_point[0]:.1f}, "
              f"{pose.surface_point[1]:.1f}, {pose.surface_point[2]:.1f}), "
              f"R=({pose.rx:.1f}, {pose.ry:.1f}, {pose.rz:.1f})")

    # 计算误差
    print("\n逐帧配准误差 (TRE 近似):")
    for i in range(len(optimal)):
        tre = np.linalg.norm(optimal[i].surface_point - true_poses[i].surface_point)
        rot_err = np.linalg.norm(optimal[i].rotation - true_poses[i].rotation)
        print(f"  帧 {i + 1}: 位置误差 = {tre:.2f} mm, 旋转误差 = {rot_err:.2f} 度")


if __name__ == "__main__":
    demo_synthetic()
    #demo_hmm_viterbi()
