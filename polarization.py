from raysect.core import World, Node, translate, rotate, Point3D, Vector3D
from raysect.optical import Material
from raysect.primitive import Box, Sphere
# Lambert 클래스는 raysect.optical.material.lambert 모듈에 있습니다
from raysect.optical.material.lambert import Lambert
from raysect.optical.material import AbsorbingSurface
from raysect.optical.observer import PinholeCamera
from raysect.optical.observer import RGBPipeline2D
from raysect.optical.library import schott
from raysect.optical.material import Conductor, RoughConductor, Dielectric
# SpectralFunction 관련 모듈 추가
from raysect.optical.spectralfunction import ConstantSF, InterpolatedSF
from raysect.optical.observer import PowerPipeline2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import warnings

# 복소수 경고 무시
warnings.filterwarnings('ignore', category=np.ComplexWarning)

# Raysect 관련 전역 변수 설정
RAYSECT_AVAILABLE = False
try:
    from raysect.optical import Point3D
    RAYSECT_AVAILABLE = True
except ImportError:
    # Raysect가 없을 경우 대체 Point3D 클래스 정의
    class Point3D:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            
        def __getitem__(self, idx):
            if idx == 0: return self.x
            elif idx == 1: return self.y
            elif idx == 2: return self.z
            raise IndexError("Point3D index out of range")

"""
초단초점 프로젝터의 광학 시뮬레이션 코드 (Optical Simulation for Ultra-Short-Throw Projector)

이 시뮬레이션의 목적:
- 초단초점 프로젝터에서 발생하는 색상 편차와 밝기 불균일성 현상을 시뮬레이션
- 편광 효과가 색상 및 밝기 분포에 미치는 영향 분석
- 반파장판(Half-Wave Plate, HWP)을 이용한 보정 방법 검증

주요 광학 개념:
1. 편광 (Polarization):
   - TE(Transverse Electric) 모드: 수평 편광, 전기장이 입사면에 수직
   - TM(Transverse Magnetic) 모드: 수직 편광, 전기장이 입사면에 평행
   
2. 프레넬 반사 (Fresnel Reflection):
   - 빛이 서로 다른 매질 경계면에서 반사될 때, 입사각과 편광 상태에 따라 반사율이 변화
   - TE 모드는 입사각이 증가해도 반사율 변화가 적음
   - TM 모드는 브루스터 각 부근에서 반사율이 크게 감소

3. 반파장판 (Half-Wave Plate, HWP):
   - 빛의 편광 상태를 90° 회전시키는 광학 소자
   - TE→TM, TM→TE 변환 가능
   - 입사각에 따른 반사율 변화를 조절하여, 색상 균일성 향상

시뮬레이션 흐름:
1. 입력 광원 정의: RGB 파장별 초기 편광 상태 설정
2. 광학 요소 모델링: DMD, 렌즈, 접이식 거울, 스크린 등
3. 편광 추적: Jones 벡터와 Jones 행렬을 이용한 편광 상태 변화 추적
4. 각도별 분석: 입사각에 따른 반사율과 편광 효과 분석
5. 결과 시각화: 색상 비율, 밝기 분포, 편광 상태 등 시각화

결과 해석:
- RGB 색상 균형 분석: 청색 비율이 높을수록 파란 색조 편향
- 기하학적 영향: 스크린 상단은 입사각이 크고, 하단은 입사각이 작음
- HWP 효과: 편광 회전을 통해 색상 편차 보정
"""

# 한글 폰트 설정 (macOS의 경우)
font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # macOS 한글 폰트 경로 예시
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 기본 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시 문제 해결

# 1. 세계 생성 - 장면 그래프의 루트
world = World()

# 2. 광학 요소 정의 및 변환
# 2.1 DMD - 작은 거울 평면
dmd_size = 0.02  # 2 cm 크기 (예시)
# DMD 원점에 배치, 렌즈로 빛을 향하게 Y축 기준 12° 기울임
dmd = Box(
    lower=Point3D(-dmd_size/2, -dmd_size/2, 0),
    upper=Point3D(dmd_size/2, dmd_size/2, 0.001),  # 거울 역할을 하는 얇은 상자
    parent=world,
    transform=rotate(0, 12, 0)  # Y축 기준 12° 기울임 (온 상태 각도)
)

# 개선된 DMD 재질: Lambert에서 RoughConductor로 변경
# RoughConductor는 방향성 금속 반사를 더 정확하게 모델링합니다
# 이 부분에서 오류가 발생하여 주석 처리합니다
# dmd.material = RoughConductor(0.95, 0.05)  # 반사율 95%, 거칠기 0.05
# 대체 코드
try:
    dmd.material = RoughConductor(Aluminium(), 0.05)  # 알루미늄 재질, 거칠기 0.05
except (NameError, TypeError):
    print("RoughConductor 클래스를 적용할 수 없습니다. 기본 재질을 사용합니다.")

# 2.2 투사 렌즈 - 얇은 렌즈(두 표면) 또는 조리개로 단순화
lens_radius = 0.05
# 간단한 렌즈 요소를 형성하기 위해 두 구면을 back-to-back으로 배치
lens1 = Sphere(radius=0.1, parent=world, transform=translate(0, 0, 0.3))  # z축을 따라 위치
lens2 = Sphere(radius=0.1, parent=world, transform=translate(0, 0, 0.32))

# 개선된 렌즈 재질: 굴절 효과를 정확하게 모델링하기 위해 Dielectric으로 변경
# 이 부분에서도 오류가 발생할 수 있어 주석 처리합니다
# lens1.material = Dielectric(1.5)  # 유리 굴절률 1.5
# lens2.material = Dielectric(1.5)
# 대체 코드
try:
    lens1.material = Dielectric(1.5)  # 유리 굴절률 1.5
    lens2.material = Dielectric(1.5)
except (NameError, TypeError):
    print("Dielectric 클래스를 적용할 수 없습니다. 기본 재질을 사용합니다.")

# 2.3 접이식 거울 - 스크린을 향해 빛을 위쪽으로 리디렉션하는 45° 각도의 평면
# Plane 대신 얇은 Box 사용
mirror_size = 0.2  # 거울 크기
mirror_thickness = 0.001  # 평면처럼 작용하도록 매우 얇게
fold_mirror = Box(
    lower=Point3D(-mirror_size/2, -mirror_size/2, -mirror_thickness/2),
    upper=Point3D(mirror_size/2, mirror_size/2, mirror_thickness/2),
    parent=world,
    transform=translate(0, -0.1, 0.5) * rotate(45, 0, 0)  # 45° 각도로 배치, 약간 아래로 이동
)

# 개선된 거울 재질: 금속의 정확한 반사 특성을 위해 RoughConductor 사용
# fold_mirror.material = RoughConductor(0.95, 0.02)  # 반사율 95%, 거칠기 0.02
# 대체 코드
try:
    fold_mirror.material = RoughConductor(Aluminium(), 0.02)  # 알루미늄 재질, 거칠기 0.02
except (NameError, TypeError):
    print("RoughConductor 클래스를 거울에 적용할 수 없습니다. 기본 재질을 사용합니다.")

# 2.4 스크린 - 이미지가 투사되는 큰 평면
screen_size = 0.5
screen_thickness = 0.001
screen = Box(
    lower=Point3D(-screen_size/2, -screen_size/2, -screen_thickness/2),
    upper=Point3D(screen_size/2, screen_size/2, screen_thickness/2),
    parent=world,
    transform=translate(0, 1.0, 1.0)
)

# 개선된 스크린 재질: 가시성을 위해 RoughConductor보다는 Lambert를 유지하지만, 반사율 조정
# screen_reflectivity = ConstantSF(0.9)
# screen.material = Lambert(screen_reflectivity)
# 대체 코드
try:
    screen_reflectivity = ConstantSF(0.9)
    screen.material = Lambert(screen_reflectivity)
except (NameError, TypeError):
    print("Lambert 클래스를 스크린에 적용할 수 없습니다. 기본 재질을 사용합니다.")

# 2.5 Z 평면 스크린 - 수직 방향 스크린 (Y-Z 평면)
z_screen_size = 0.5
z_screen_thickness = 0.001
z_screen = Box(
    lower=Point3D(-z_screen_thickness/2, -z_screen_size/2, -z_screen_size/2),
    upper=Point3D(z_screen_thickness/2, z_screen_size/2, z_screen_size/2),
    parent=world,
    transform=translate(1.0, 1.0, 1.0)
)
# Z 스크린에도 동일한 재질 적용
try:
    z_screen.material = Lambert(screen_reflectivity)
except (NameError, TypeError):
    print("Lambert 클래스를 Z 스크린에 적용할 수 없습니다. 기본 재질을 사용합니다.")

# 3. 편광 분석을 위한 Jones 벡터 설정
# 수평(TE) 및 수직(TM) 선형 편광을 위한 Jones 벡터
jones_H = np.array([1+0j, 0+0j])  # 수평 단위 편광 (TE)
jones_V = np.array([0+0j, 1+0j])  # 수직 단위 편광 (TM)

# 각 색상에 초기 편광 할당
pol_red = jones_V.copy()    # 빨간색 644nm: TM (수직)
pol_green = jones_H.copy()  # 녹색 550nm: TE (수평)
pol_blue = jones_H.copy()   # 파란색 460nm: TE (수평)

# 각 색상의 파장(nm) 정의
wavelength_red = 644
wavelength_green = 550
wavelength_blue = 460

# 4. 접이식 거울에 대한 프레넬 반사 계수 정의 (단순화된 모델)
def fresnel_coeffs(angle_deg, wavelength_nm=None):
    """
    주어진 입사각과 파장에 대한 (r_s, r_p) 진폭 반사 계수 반환
    
    파라미터:
    - angle_deg: 입사각 (도)
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=None
    
    반환값:
    - r_s: s-편광(TE 모드)에 대한 반사 계수
    - r_p: p-편광(TM 모드)에 대한 반사 계수
    
    설명:
    실제 금속 반사면에 대한 프레넬 반사 계수를 계산합니다.
    s-편광(TE)은 입사각에 따라 반사율이 비교적 일정하게 유지되는 반면,
    p-편광(TM)은 브루스터 각 부근에서 반사율이 크게 감소합니다.
    """
    # 복소수 굴절률: 알루미늄 코팅 거울 가정
    # 알루미늄의 파장별 복소 굴절률 데이터
    n1 = 1.0      # 공기
    
    # 알루미늄의 파장별 복소 굴절률 데이터
    n2_data = {
        400: 0.63 + 4.82j,    # 400nm (파란색 근처)
        460: 0.68 + 5.32j,    # 460nm (파란색)
        550: 0.96 + 6.69j,    # 550nm (녹색)
        644: 1.15 + 7.84j,    # 644nm (빨간색)
        700: 1.23 + 8.12j     # 700nm (깊은 빨간색 근처)
    }
    
    # 파장 값이 제공되었는지 확인
    if wavelength_nm is None:
        # 파장이 제공되지 않으면 550nm(녹색) 기준 값 사용
        wavelength_nm = 550
    
    # 제공된 파장에 대한 굴절률 결정
    if wavelength_nm in n2_data:
        n2 = n2_data[wavelength_nm]
    else:
        # 파장이 데이터에 없는 경우 선형 보간 적용
        wl_keys = list(n2_data.keys())
        wl_keys.sort()
        
        if wavelength_nm < wl_keys[0]:
            # 가장 작은 파장보다 작은 경우
            n2 = n2_data[wl_keys[0]]
        elif wavelength_nm > wl_keys[-1]:
            # 가장 큰 파장보다 큰 경우
            n2 = n2_data[wl_keys[-1]]
        else:
            # 중간 파장인 경우 선형 보간
            for i in range(len(wl_keys)-1):
                if wl_keys[i] <= wavelength_nm <= wl_keys[i+1]:
                    w1, w2 = wl_keys[i], wl_keys[i+1]
                    n2_1, n2_2 = n2_data[w1], n2_data[w2]
                    weight = (wavelength_nm - w1) / (w2 - w1)
                    real_part = n2_1.real * (1-weight) + n2_2.real * weight
                    imag_part = n2_1.imag * (1-weight) + n2_2.imag * weight
                    n2 = complex(real_part, imag_part)
                    break
    
    # 라디안으로 변환
    theta_i = np.deg2rad(angle_deg)
    
    # 스넬의 법칙 (복소수 굴절률을 사용한 각도 계산)
    # sin(theta_t) = n1/n2 * sin(theta_i)
    sin_theta_t = n1/n2 * np.sin(theta_i)
    cos_theta_i = np.cos(theta_i)
    
    # 복소수 코사인 계산
    cos_theta_t = np.sqrt(1 - sin_theta_t**2 + 0j)  # 복소수 결과를 보장하기 위해 0j 추가
    
    # 프레넬 반사 계수 (r_s, r_p)
    # s-편광 (전기장이 입사면에 수직, TE)
    r_s = (n1*cos_theta_i - n2*cos_theta_t) / (n1*cos_theta_i + n2*cos_theta_t)
    
    # p-편광 (전기장이 입사면에 평행, TM)
    r_p = (n2*cos_theta_i - n1*cos_theta_t) / (n2*cos_theta_i + n1*cos_theta_t)
    
    # 복소수 반사 계수를 직접 반환
    # 이전에는 절대값을 반환했지만, 이제 위상 정보도 포함
    return r_s, r_p

# 편광 상태 시각화 함수 추가
def plot_polarization_state(jones_vector, ax, title):
    """
    Jones 벡터의 편광 상태를 시각화하는 함수
    """
    # 벡터 성분
    Ex_real = np.real(jones_vector[0])
    Ex_imag = np.imag(jones_vector[0])
    Ey_real = np.real(jones_vector[1])
    Ey_imag = np.imag(jones_vector[1])
    
    # 벡터 크기 계산
    Ex_amp = np.sqrt(Ex_real**2 + Ex_imag**2)
    Ey_amp = np.sqrt(Ey_real**2 + Ey_imag**2)
    
    # 위상 계산
    Ex_phase = np.arctan2(Ex_imag, Ex_real)
    Ey_phase = np.arctan2(Ey_imag, Ey_real)
    
    # 타원 매개변수
    t = np.linspace(0, 2*np.pi, 100)
    E_x = Ex_amp * np.cos(t + Ex_phase)
    E_y = Ey_amp * np.cos(t + Ey_phase)
    
    # 플롯
    ax.plot(E_x, E_y)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Ex')
    ax.set_ylabel('Ey')
    ax.set_title(title)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 수평(TE)과 수직(TM) 방향 표시
    ax.text(1.3, 0.1, 'TE (수평)', fontsize=8)
    ax.text(0.1, 1.3, 'TM (수직)', fontsize=8)
    
    # 편광 타입 결정 및 표시
    delta_phase = Ey_phase - Ex_phase
    if np.isclose(Ey_amp, 0, atol=1e-10) or np.isclose(Ex_amp, 0, atol=1e-10):
        pol_type = "선형 편광"
    elif np.isclose(Ex_amp, Ey_amp, atol=1e-10):
        if np.isclose(np.abs(delta_phase), np.pi/2, atol=1e-10):
            pol_type = "원형 편광"
        else:
            pol_type = "타원 편광"
    else:
        pol_type = "타원 편광"
    
    ax.text(-1.4, -1.3, f"편광 유형: {pol_type}", fontsize=8)
    
    return ax

# 5. 광선 전파 함수 (HWP 없음)
def propagate_without_hwp(jones_in, incidence_angle, wavelength_nm=550):
    """
    광선의 편광 상태를 전파하는 함수 (HWP 없음)
    
    파라미터:
    - jones_in: 입력 Jones 벡터
    - incidence_angle: 입사각 (도)
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=550 (녹색)
    
    반환값:
    - 전파 후 Jones 벡터
    """
    # 1. DMD (편광 변화 없이 단순 반사 가정)
    jones_after_dmd = jones_in.copy()
    # 2. 투사 렌즈 (편광 변화 없음)
    jones_after_lens = jones_after_dmd
    # 3. 접이식 거울: 각도 의존적 Jones 행렬 적용
    r_s, r_p = fresnel_coeffs(incidence_angle, wavelength_nm)
    # H/V 기준에서 H = s, V = p:
    J_mirror = np.array([[r_s, 0],
                         [0, r_p]], dtype=complex)
    jones_after_mirror = J_mirror.dot(jones_after_lens)
    # 4. 스크린 (확산) - Jones 벡터에서 강도 추출 (스크린 이후 편광은 더 이상 필요없음)
    return jones_after_mirror

# 6. 광선 전파 함수 (HWP 포함)
def propagate_with_hwp(jones_in, incidence_angle, wavelength_nm=550):
    """
    광선의 편광 상태를 전파하는 함수 (HWP 포함)
    
    파라미터:
    - jones_in: 입력 Jones 벡터
    - incidence_angle: 입사각 (도)
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=550 (녹색)
    
    반환값:
    - 전파 후 Jones 벡터
    """
    # 렌즈까지는 이전과 동일
    jones_after_dmd = jones_in.copy()
    jones_after_lens = jones_after_dmd
    # 반파장판(λ/2) 삽입 - 수직을 수평으로 회전시키도록 배향
    # HWP 축이 수직에서 45° 기울어짐 가정
    # 45°에서의 HWP Jones 행렬: 기본적으로 H<->V 교환
    J_hwp = np.array([[0, 1],
                      [1, 0]], dtype=complex)  # 이상적인 45° 반파장판 (편광을 90° 회전)
    jones_after_hwp = J_hwp.dot(jones_after_lens)
    # 이제 접이식 거울 통과:
    r_s, r_p = fresnel_coeffs(incidence_angle, wavelength_nm)
    J_mirror = np.array([[r_s, 0],
                         [0, r_p]], dtype=complex)
    jones_after_mirror = J_mirror.dot(jones_after_hwp)
    return jones_after_mirror

# 6-S. Green과 Blue에만 HWP를 적용하는 선택적 함수
def propagate_with_selective_hwp(jones_in, incidence_angle, color_name, wavelength_nm=None):
    """
    Green과 Blue에만 HWP를 적용하고, Red에는 적용하지 않는 함수
    
    파라미터:
    - jones_in: 입력 Jones 벡터
    - incidence_angle: 입사각 (도)
    - color_name: 색상 이름 ('Red', 'Green', 'Blue')
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=None (색상 이름에 따라 자동 결정)
    
    반환값:
    - Jones 벡터
    """
    # 색상 이름에 따라 파장 결정 (파장이 명시적으로 제공되지 않은 경우)
    if wavelength_nm is None:
        if color_name == 'Red':
            wavelength_nm = 644
        elif color_name == 'Green':
            wavelength_nm = 550
        elif color_name == 'Blue':
            wavelength_nm = 460
        else:
            wavelength_nm = 550  # 기본값
    
    # 렌즈까지는 이전과 동일
    jones_after_dmd = jones_in.copy()
    jones_after_lens = jones_after_dmd
    
    # Green과 Blue에만 HWP 적용
    if color_name in ['Green', 'Blue']:
        # HWP 적용
        J_hwp = np.array([[0, 1],
                          [1, 0]], dtype=complex)
        jones_after_lens = J_hwp.dot(jones_after_lens)
    
    # Red는 HWP 적용 없이 그대로 통과
    
    # 접이식 거울로 반사
    r_s, r_p = fresnel_coeffs(incidence_angle, wavelength_nm)
    J_mirror = np.array([[r_s, 0],
                         [0, r_p]], dtype=complex)
    jones_after_mirror = J_mirror.dot(jones_after_lens)
    return jones_after_mirror

# 6-Z. Z 평면 스크린을 위한 광선 전파 함수 (HWP 없음)
def propagate_without_hwp_z(jones_in, incidence_angle, wavelength_nm=550):
    """
    Z 평면 스크린을 위한 광선 전파 함수 (HWP 없음)
    
    파라미터:
    - jones_in: 입력 Jones 벡터
    - incidence_angle: 입사각 (도)
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=550 (녹색)
    
    반환값:
    - 전파 후 Jones 벡터
    """
    # X-Y 스크린과 동일한 전파 과정이지만, 편광 방향이 바뀔 수 있음
    # Z 평면 스크린에서는 p-편광이 이제 수평, s-편광이 수직 방향
    jones_after_dmd = jones_in.copy()
    jones_after_lens = jones_after_dmd
    
    # Z 스크린에 대한 반사는 X-Y 평면 스크린과 다른 편광 매핑
    # X-Y 스크린: TE(수평)=s, TM(수직)=p
    # Z 스크린: TE(수직)=s, TM(수평)=p (90도 회전)
    # 편광 90° 회전 행렬
    J_rotate = np.array([[0, 1],
                         [1, 0]], dtype=complex)
    jones_rotated = J_rotate.dot(jones_after_lens)
    
    # 접이식 거울로 반사
    r_s, r_p = fresnel_coeffs(incidence_angle, wavelength_nm)
    J_mirror = np.array([[r_s, 0],
                         [0, r_p]], dtype=complex)
    jones_after_mirror = J_mirror.dot(jones_rotated)
    return jones_after_mirror

# 6-Z2. Z 평면 스크린을 위한 광선 전파 함수 (HWP 포함)
def propagate_with_hwp_z(jones_in, incidence_angle, wavelength_nm=550):
    """
    Z 평면 스크린을 위한 광선 전파 함수 (HWP 포함)
    
    파라미터:
    - jones_in: 입력 Jones 벡터
    - incidence_angle: 입사각 (도)
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=550 (녹색)
    
    반환값:
    - 전파 후 Jones 벡터
    """
    # 렌즈까지는 동일
    jones_after_dmd = jones_in.copy()
    jones_after_lens = jones_after_dmd
    
    # HWP 적용
    J_hwp = np.array([[0, 1],
                      [1, 0]], dtype=complex)
    jones_after_hwp = J_hwp.dot(jones_after_lens)
    
    # Z 스크린에 대한 편광 회전
    J_rotate = np.array([[0, 1],
                         [1, 0]], dtype=complex)
    jones_rotated = J_rotate.dot(jones_after_hwp)
    
    # 접이식 거울로 반사
    r_s, r_p = fresnel_coeffs(incidence_angle, wavelength_nm)
    J_mirror = np.array([[r_s, 0],
                         [0, r_p]], dtype=complex)
    jones_after_mirror = J_mirror.dot(jones_rotated)
    return jones_after_mirror

# 6-Z3. Z 평면 스크린을 위한 선택적 HWP 적용 함수
def propagate_with_selective_hwp_z(jones_in, incidence_angle, color_name, wavelength_nm=None):
    """
    Z 스크린에 대해 Green과 Blue에만 HWP를 적용하는 함수
    
    파라미터:
    - jones_in: 입력 Jones 벡터
    - incidence_angle: 입사각 (도)
    - color_name: 색상 이름 ('Red', 'Green', 'Blue')
    - wavelength_nm: 빛의 파장 (나노미터), 기본값=None (색상 이름에 따라 자동 결정)
    
    반환값:
    - 전파 후 Jones 벡터
    """
    # 색상 이름에 따라 파장 결정 (파장이 명시적으로 제공되지 않은 경우)
    if wavelength_nm is None:
        if color_name == 'Red':
            wavelength_nm = 644
        elif color_name == 'Green':
            wavelength_nm = 550
        elif color_name == 'Blue':
            wavelength_nm = 460
        else:
            wavelength_nm = 550  # 기본값
    
    # 렌즈까지는 동일
    jones_after_dmd = jones_in.copy()
    jones_after_lens = jones_after_dmd
    
    # Green과 Blue에만 HWP 적용
    if color_name in ['Green', 'Blue']:
        # HWP 적용
        J_hwp = np.array([[0, 1],
                          [1, 0]], dtype=complex)
        jones_after_lens = J_hwp.dot(jones_after_lens)
    
    # Z 스크린에 대한 편광 회전
    J_rotate = np.array([[0, 1],
                         [1, 0]], dtype=complex)
    jones_rotated = J_rotate.dot(jones_after_lens)
    
    # 접이식 거울로 반사
    r_s, r_p = fresnel_coeffs(incidence_angle, wavelength_nm)
    J_mirror = np.array([[r_s, 0],
                         [0, r_p]], dtype=complex)
    jones_after_mirror = J_mirror.dot(jones_rotated)
    return jones_after_mirror

# 7. 샘플 각도에서 각 파장에 대한 전파 테스트
print("\n--- 편광 분석 결과 ---")
angles = [10, 20, 30, 40, 50, 60, 70, 80]  # 입사각 (도) - 10도부터 80도까지 10도 간격
colors = {"Red": pol_red, "Green": pol_green, "Blue": pol_blue}
# 실제 색상 매핑
color_map = {"Red": 'red', "Green": 'green', "Blue": 'blue'}

# 표 제목 업데이트 - HWP 적용 방식 명시
print("입사각도 | 색상  | HWP 전 강도 | GB만 HWP 적용 강도")
print("---------|-------|------------|------------------")
for angle in angles:
    for color, jones_vec in colors.items():
        out_no_hwp = propagate_without_hwp(jones_vec, angle)
        # 선택적 HWP 적용 (Green과 Blue만)
        out_selective_hwp = propagate_with_selective_hwp(jones_vec, angle, color)
        
        # 강도는 |E_x|^2 + |E_y|^2
        I_no_hwp = np.abs(out_no_hwp[0])**2 + np.abs(out_no_hwp[1])**2
        I_selective_hwp = np.abs(out_selective_hwp[0])**2 + np.abs(out_selective_hwp[1])**2
        
        print(f"{angle:>6}° | {color:5} | {I_no_hwp:6.3f}     | {I_selective_hwp:6.3f}")
    print("---------|-------|------------|------------------")

# 8. 스크린의 색상 편차 및 밝기 불균일성 수치 매핑
Ny, Nx = 50, 50
y_vals = np.linspace(-1, 1, Ny)  # -1 = 하단, +1 = 상단 (상대적 스크린)
x_vals = np.linspace(-1, 1, Nx)  # -1 = 왼쪽, +1 = 오른쪽
# 색상 강도를 위한 배열 생성
intensity_map_before = np.zeros((Ny, Nx, 3))  # 보정 전 RGB 강도
intensity_map_after = np.zeros((Ny, Nx, 3))   # 선택적 HWP 적용 후 RGB 강도

# Z 스크린을 위한 강도 맵
intensity_map_z_before = np.zeros((Ny, Nx, 3))  # Z 스크린 보정 전 RGB 강도
intensity_map_z_after = np.zeros((Ny, Nx, 3))   # Z 스크린 선택적 HWP 적용 후 RGB 강도

# 스크린 위치별 편광 및 강도 계산
for i, y in enumerate(y_vals):
    # y를 거울의 입사각에 매핑 (중앙 y=0 -> 45°, 상단 y=+1 -> ~80°, 하단 y=-1 -> ~10°)
    inc_angle = 45 + (y * 35)  # 수정된 매핑 (10도~80도 범위를 포함하도록)
    for j, x in enumerate(x_vals):
        # 단순 모델에서는 수평 변화가 입사각을 크게 변화시키지 않는다고 가정
        for c, (color_name, jones_vec) in enumerate(list(colors.items())):
            # RGB 순서: Red=0, Green=1, Blue=2
            out_no_hwp = propagate_without_hwp(jones_vec, inc_angle)
            # 선택적 HWP 적용
            out_selective_hwp = propagate_with_selective_hwp(jones_vec, inc_angle, color_name)
            
            intensity_map_before[i, j, c] = np.abs(out_no_hwp[0])**2 + np.abs(out_no_hwp[1])**2
            intensity_map_after[i, j, c] = np.abs(out_selective_hwp[0])**2 + np.abs(out_selective_hwp[1])**2
            
            # Z 평면 스크린에 대한 계산
            out_no_hwp_z = propagate_without_hwp_z(jones_vec, inc_angle)
            # Z 스크린에 대한 선택적 HWP 적용
            out_selective_hwp_z = propagate_with_selective_hwp_z(jones_vec, inc_angle, color_name)
            
            intensity_map_z_before[i, j, c] = np.abs(out_no_hwp_z[0])**2 + np.abs(out_no_hwp_z[1])**2
            intensity_map_z_after[i, j, c] = np.abs(out_selective_hwp_z[0])**2 + np.abs(out_selective_hwp_z[1])**2

# 9. 휘도(밝기) 및 색상 편향 계산
brightness_before = intensity_map_before.sum(axis=2)
brightness_after = intensity_map_after.sum(axis=2)
blue_ratio_before = intensity_map_before[:,:,2] / (intensity_map_before.sum(axis=2) + 1e-8)
blue_ratio_after = intensity_map_after[:,:,2] / (intensity_map_after.sum(axis=2) + 1e-8)

# Z 스크린에 대한 휘도 및 색상 편향 계산
brightness_z_before = intensity_map_z_before.sum(axis=2)
brightness_z_after = intensity_map_z_after.sum(axis=2)
blue_ratio_z_before = intensity_map_z_before[:,:,2] / (intensity_map_z_before.sum(axis=2) + 1e-8)
blue_ratio_z_after = intensity_map_z_after[:,:,2] / (intensity_map_z_after.sum(axis=2) + 1e-8)

# 10. 결과 시각화
# 색상 더 직관적인 컬러맵 설정
blue_cmap = plt.cm.Blues
red_cmap = plt.cm.Reds
green_cmap = plt.cm.Greens

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 파란색 강도 비율 시각화 - 더 직관적인 청색 컬러맵 사용
im0 = axs[0, 0].imshow(blue_ratio_before, origin='lower', extent=[-1,1,-1,1], cmap='Blues', 
                      vmin=0.25, vmax=0.35)
axs[0, 0].set_title("청색 강도 비율 – HWP 적용 전")
axs[0, 0].set_xlabel("스크린 X"); axs[0, 0].set_ylabel("스크린 Y")
plt.colorbar(im0, ax=axs[0, 0])

im1 = axs[0, 1].imshow(blue_ratio_after, origin='lower', extent=[-1,1,-1,1], cmap='Blues',
                      vmin=0.25, vmax=0.35)
axs[0, 1].set_title("청색 강도 비율 – GB만 HWP 적용")
axs[0, 1].set_xlabel("스크린 X"); axs[0, 1].set_ylabel("스크린 Y")
plt.colorbar(im1, ax=axs[0, 1])

# 전체 밝기 시각화
im2 = axs[1, 0].imshow(brightness_before, origin='lower', extent=[-1,1,-1,1], cmap='viridis')
axs[1, 0].set_title("전체 밝기 – HWP 적용 전")
axs[1, 0].set_xlabel("스크린 X"); axs[1, 0].set_ylabel("스크린 Y")
plt.colorbar(im2, ax=axs[1, 0])

im3 = axs[1, 1].imshow(brightness_after, origin='lower', extent=[-1,1,-1,1], cmap='viridis')
axs[1, 1].set_title("전체 밝기 – GB만 HWP 적용")
axs[1, 1].set_xlabel("스크린 X"); axs[1, 1].set_ylabel("스크린 Y")
plt.colorbar(im3, ax=axs[1, 1])

plt.tight_layout()
plt.savefig('color_brightness_comparison_selective.png')
plt.show()

# 각 색상 비율을 별도로 시각화
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Red 비율 - 적색 컬러맵 사용
red_ratio_before = intensity_map_before[:,:,0] / (intensity_map_before.sum(axis=2) + 1e-8)
red_ratio_after = intensity_map_after[:,:,0] / (intensity_map_after.sum(axis=2) + 1e-8)

im_red1 = axs[0, 0].imshow(red_ratio_before, origin='lower', extent=[-1,1,-1,1], cmap='Reds', 
                          vmin=0.25, vmax=0.35)
axs[0, 0].set_title("적색 강도 비율 – HWP 적용 전")
axs[0, 0].set_xlabel("스크린 X"); axs[0, 0].set_ylabel("스크린 Y")
plt.colorbar(im_red1, ax=axs[0, 0])

im_red2 = axs[1, 0].imshow(red_ratio_after, origin='lower', extent=[-1,1,-1,1], cmap='Reds',
                          vmin=0.25, vmax=0.35)
axs[1, 0].set_title("적색 강도 비율 – GB만 HWP 적용")
axs[1, 0].set_xlabel("스크린 X"); axs[1, 0].set_ylabel("스크린 Y")
plt.colorbar(im_red2, ax=axs[1, 0])

# Green 비율 - 녹색 컬러맵 사용
green_ratio_before = intensity_map_before[:,:,1] / (intensity_map_before.sum(axis=2) + 1e-8)
green_ratio_after = intensity_map_after[:,:,1] / (intensity_map_after.sum(axis=2) + 1e-8)

im_green1 = axs[0, 1].imshow(green_ratio_before, origin='lower', extent=[-1,1,-1,1], cmap='Greens', 
                            vmin=0.25, vmax=0.35)
axs[0, 1].set_title("녹색 강도 비율 – HWP 적용 전")
axs[0, 1].set_xlabel("스크린 X"); axs[0, 1].set_ylabel("스크린 Y")
plt.colorbar(im_green1, ax=axs[0, 1])

im_green2 = axs[1, 1].imshow(green_ratio_after, origin='lower', extent=[-1,1,-1,1], cmap='Greens',
                            vmin=0.25, vmax=0.35)
axs[1, 1].set_title("녹색 강도 비율 – GB만 HWP 적용")
axs[1, 1].set_xlabel("스크린 X"); axs[1, 1].set_ylabel("스크린 Y")
plt.colorbar(im_green2, ax=axs[1, 1])

# Blue 비율 - 청색 컬러맵 사용
im_blue1 = axs[0, 2].imshow(blue_ratio_before, origin='lower', extent=[-1,1,-1,1], cmap='Blues', 
                           vmin=0.25, vmax=0.35)
axs[0, 2].set_title("청색 강도 비율 – HWP 적용 전")
axs[0, 2].set_xlabel("스크린 X"); axs[0, 2].set_ylabel("스크린 Y")
plt.colorbar(im_blue1, ax=axs[0, 2])

im_blue2 = axs[1, 2].imshow(blue_ratio_after, origin='lower', extent=[-1,1,-1,1], cmap='Blues',
                           vmin=0.25, vmax=0.35)
axs[1, 2].set_title("청색 강도 비율 – GB만 HWP 적용")
axs[1, 2].set_xlabel("스크린 X"); axs[1, 2].set_ylabel("스크린 Y")
plt.colorbar(im_blue2, ax=axs[1, 2])

plt.tight_layout()
plt.savefig('rgb_ratios_comparison_selective.png')
plt.show()

# 샘플 입사각에서 편광 상태의 변화 시각화
sample_angle = 60  # 60도 입사각에서의 편광 상태 변화 시각화
fig, axs = plt.subplots(3, 3, figsize=(15, 12))

# 각 색상에 대한 초기 편광 상태
plot_polarization_state(pol_red, axs[0, 0], f'RED 초기 편광 (TM)')
plot_polarization_state(pol_green, axs[1, 0], f'GREEN 초기 편광 (TE)')
plot_polarization_state(pol_blue, axs[2, 0], f'BLUE 초기 편광 (TE)')

# HWP 적용 전 편광 상태
red_no_hwp = propagate_without_hwp(pol_red, sample_angle)
green_no_hwp = propagate_without_hwp(pol_green, sample_angle)
blue_no_hwp = propagate_without_hwp(pol_blue, sample_angle)

plot_polarization_state(red_no_hwp, axs[0, 1], f'RED HWP 전 ({sample_angle}°)')
plot_polarization_state(green_no_hwp, axs[1, 1], f'GREEN HWP 전 ({sample_angle}°)')
plot_polarization_state(blue_no_hwp, axs[2, 1], f'BLUE HWP 전 ({sample_angle}°)')

# 선택적 HWP 적용 후 편광 상태 (Green, Blue만 HWP 적용)
red_selective_hwp = propagate_with_selective_hwp(pol_red, sample_angle, 'Red')
green_selective_hwp = propagate_with_selective_hwp(pol_green, sample_angle, 'Green')
blue_selective_hwp = propagate_with_selective_hwp(pol_blue, sample_angle, 'Blue')

plot_polarization_state(red_selective_hwp, axs[0, 2], f'RED - HWP 없음 ({sample_angle}°)')
plot_polarization_state(green_selective_hwp, axs[1, 2], f'GREEN - HWP 적용 ({sample_angle}°)')
plot_polarization_state(blue_selective_hwp, axs[2, 2], f'BLUE - HWP 적용 ({sample_angle}°)')

plt.tight_layout()
plt.savefig('polarization_states_selective.png')
plt.show()

# 입사각과 강도의 관계를 시각화하는 그래프 추가
plt.figure(figsize=(10, 6))
intensity_before = np.zeros((len(angles), 3))
intensity_after = np.zeros((len(angles), 3))

# 각 각도별 강도 계산
for i, angle in enumerate(angles):
    for c, (color_name, jones_vec) in enumerate(colors.items()):
        out_no_hwp = propagate_without_hwp(jones_vec, angle)
        out_selective_hwp = propagate_with_selective_hwp(jones_vec, angle, color_name)
        intensity_before[i, c] = np.abs(out_no_hwp[0])**2 + np.abs(out_no_hwp[1])**2
        intensity_after[i, c] = np.abs(out_selective_hwp[0])**2 + np.abs(out_selective_hwp[1])**2

# HWP 적용 전 그래프
for c, (color_name, _) in enumerate(colors.items()):
    plt.plot(angles, intensity_before[:, c], 
             color=color_map[color_name], 
             linestyle='--', 
             marker='o',
             label=f'{color_name} (HWP 전)')

# 선택적 HWP 적용 후 그래프
for c, (color_name, _) in enumerate(colors.items()):
    hwp_status = "HWP 없음" if color_name == "Red" else "HWP 적용"
    plt.plot(angles, intensity_after[:, c], 
             color=color_map[color_name], 
             linestyle='-', 
             marker='s',
             label=f'{color_name} ({hwp_status})')

plt.title('입사각에 따른 색상별 강도 변화 (Green/Blue만 HWP 적용)')
plt.xlabel('입사각 (도)')
plt.ylabel('강도')
plt.grid(True)
plt.legend()
plt.savefig('angle_intensity_relation_selective.png')
plt.show()

# 조합된 RGB 색상 효과 시각화를 위한 추가 그래프
plt.figure(figsize=(12, 6))
center_line = Nx // 2

# 색상 균형 비교
plt.subplot(1, 2, 1)
red_ratio_before = intensity_map_before[:, center_line, 0] / intensity_map_before[:, center_line].sum(axis=1)
green_ratio_before = intensity_map_before[:, center_line, 1] / intensity_map_before[:, center_line].sum(axis=1)
blue_ratio_before = intensity_map_before[:, center_line, 2] / intensity_map_before[:, center_line].sum(axis=1)

red_ratio_after = intensity_map_after[:, center_line, 0] / intensity_map_after[:, center_line].sum(axis=1)
green_ratio_after = intensity_map_after[:, center_line, 1] / intensity_map_after[:, center_line].sum(axis=1)
blue_ratio_after = intensity_map_after[:, center_line, 2] / intensity_map_after[:, center_line].sum(axis=1)

plt.plot(y_vals, red_ratio_before, 'r--', label='Red 비율 (HWP 전)')
plt.plot(y_vals, green_ratio_before, 'g--', label='Green 비율 (HWP 전)')
plt.plot(y_vals, blue_ratio_before, 'b--', label='Blue 비율 (HWP 전)')

plt.plot(y_vals, red_ratio_after, 'r-', label='Red 비율 (GB만 HWP)')
plt.plot(y_vals, green_ratio_after, 'g-', label='Green 비율 (GB만 HWP)')
plt.plot(y_vals, blue_ratio_after, 'b-', label='Blue 비율 (GB만 HWP)')

plt.axhline(y=1/3, color='k', linestyle=':', label='이상적인 비율 (1/3)')
plt.title('색상 균형 비교 (Green/Blue만 HWP 적용)')
plt.xlabel('스크린 Y 위치')
plt.ylabel('색상 비율')
plt.grid(True)
plt.legend()

# RGB 통합 밝기 변화
plt.subplot(1, 2, 2)
total_before = intensity_map_before[:, center_line].sum(axis=1)
total_after = intensity_map_after[:, center_line].sum(axis=1)

plt.plot(y_vals, total_before, 'k--', label='전체 밝기 (HWP 전)')
plt.plot(y_vals, total_after, 'k-', label='전체 밝기 (GB만 HWP)')
plt.title('전체 밝기 변화 (Green/Blue만 HWP 적용)')
plt.xlabel('스크린 Y 위치')
plt.ylabel('합산 강도')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('rgb_combined_effect_selective.png')
plt.show()

# RGB 색상 분포 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 정규화 및 감마 보정을 적용하여 더 자연스러운 색상 표현
rgb_before = intensity_map_before / np.max(intensity_map_before)
# 감마 보정 (gamma = 2.2)
rgb_before = rgb_before ** (1/2.2)
plt.imshow(rgb_before, origin='lower', extent=[-1,1,-1,1])
plt.title('시뮬레이션된 화면 색상 - HWP 적용 전')
plt.xlabel('스크린 X'); plt.ylabel('스크린 Y')

plt.subplot(1, 2, 2)
rgb_after = intensity_map_after / np.max(intensity_map_after)
# 감마 보정 적용
rgb_after = rgb_after ** (1/2.2)
plt.imshow(rgb_after, origin='lower', extent=[-1,1,-1,1])
plt.title('시뮬레이션된 화면 색상 - Green/Blue만 HWP 적용')
plt.xlabel('스크린 X'); plt.ylabel('스크린 Y')

plt.tight_layout()
plt.savefig('color_simulation_selective.png')
plt.show()

# 입사각에 따른 TE/TM 모드 반사율 비교 그래프 추가
plt.figure(figsize=(10, 6))
angles_detailed = np.arange(0, 90, 1)  # 0도부터 89도까지 1도 간격
rs_values = []  # TE 모드(s-편광) 반사율
rp_values = []  # TM 모드(p-편광) 반사율

for angle in angles_detailed:
    r_s, r_p = fresnel_coeffs(angle)
    rs_values.append(r_s**2)  # 에너지 반사율 = |r|^2
    rp_values.append(r_p**2)

plt.plot(angles_detailed, rs_values, 'b-', label='TE 모드(s-편광) 반사율')
plt.plot(angles_detailed, rp_values, 'r-', label='TM 모드(p-편광) 반사율')
plt.axvline(x=56.5, color='k', linestyle='--', alpha=0.5, label='브루스터 각 (약 56.5°)')

plt.title('입사각에 따른 TE/TM 모드 반사율 비교')
plt.xlabel('입사각 (도)')
plt.ylabel('반사율')
plt.grid(True)
plt.legend()
plt.savefig('te_tm_reflection_comparison.png')
plt.show()

# 입사각에 따른 투과율 그래프 (반사율에서 유도)
plt.figure(figsize=(10, 6))
ts_values = [1 - rs for rs in rs_values]  # 투과율 = 1 - 반사율 (단순화 모델)
tp_values = [1 - rp for rp in rp_values]

plt.plot(angles_detailed, ts_values, 'b-', label='TE 모드(s-편광) 투과율')
plt.plot(angles_detailed, tp_values, 'r-', label='TM 모드(p-편광) 투과율')
plt.axvline(x=56.5, color='k', linestyle='--', alpha=0.5, label='브루스터 각 (약 56.5°)')

plt.title('입사각에 따른 TE/TM 모드 투과율 비교')
plt.xlabel('입사각 (도)')
plt.ylabel('투과율')
plt.grid(True)
plt.legend()
plt.savefig('te_tm_transmission_comparison.png')
plt.show()

# 프로젝터-스크린 기하학 구조 시각화 (2D)
plt.figure(figsize=(10, 8))

# 컴포넌트 위치 설정
dmd_width = 0.4
dmd_height = 0.2
dmd_x = -1.0
dmd_y = 0.0

lens_x = -0.5
lens_y = 0.0
lens_radius = 0.15

mirror_center_x = 0.5
mirror_center_y = 0.0
mirror_radius = 1.2
mirror_start_angle = -60  # 도
mirror_end_angle = 0  # 도

screen_height = 1.0
screen_x = -0.5
screen_top_y = 1.2

# 스크린 그리기 (수직선)
plt.plot([screen_x, screen_x], [screen_top_y-screen_height, screen_top_y], 'k-', linewidth=2)
plt.text(screen_x-0.15, screen_top_y+0.05, '스크린', fontsize=12)

# DMD 패널 그리기
dmd_rect = plt.Rectangle((dmd_x-dmd_width/2, dmd_y-dmd_height/2), dmd_width, dmd_height, 
                        fill=True, color='lightblue', alpha=0.7)
plt.gca().add_patch(dmd_rect)
plt.text(dmd_x-dmd_width/2-0.25, dmd_y, 'DMD 패널', fontsize=12)

# 렌즈 그리기
lens_ellipse = plt.Circle((lens_x, lens_y), lens_radius, fill=True, color='skyblue', alpha=0.5)
plt.gca().add_patch(lens_ellipse)
plt.text(lens_x, lens_y-lens_radius-0.1, '렌즈', fontsize=12)

# 광학축 표시
plt.plot([dmd_x-dmd_width/2-0.2, dmd_x-dmd_width/2], [dmd_y, dmd_y], 'k--', alpha=0.5)
plt.text(dmd_x-dmd_width/2-0.25, dmd_y-0.1, 'O$_A$', fontsize=10)

# 비구면 미러 그리기 (원호 일부)
theta = np.linspace(np.deg2rad(mirror_start_angle), np.deg2rad(mirror_end_angle), 100)
x_mirror = mirror_center_x + mirror_radius * np.cos(theta)
y_mirror = mirror_center_y + mirror_radius * np.sin(theta)
plt.plot(x_mirror, y_mirror, '-', linewidth=3, color='gray')
plt.text(max(x_mirror)+0.05, y_mirror[0], '비구면 미러 표면', fontsize=12)

# 거리 A 표시
plt.arrow(dmd_x, dmd_y-0.5, mirror_center_x+mirror_radius-dmd_x, 0, 
         head_width=0.05, head_length=0.05, fc='k', ec='k')
plt.text((dmd_x + mirror_center_x+mirror_radius)/2, dmd_y-0.6, 'A', fontsize=14)

# 광선 경로 - DMD에서 스크린까지 다른 색상으로 (빨간색, 녹색, 파란색)
# 빨간색 광선 (상단)
ray_angles_red = [58, 72, 84]  # 도
for angle in ray_angles_red:
    # DMD에서 렌즈로
    plt.plot([dmd_x, lens_x], [dmd_y, dmd_y], '-', color='darkred', alpha=0.8, linewidth=1)
    
    # 렌즈에서 미러로 (각도 적용)
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    mirror_point_x = lens_x + dx * 1.0
    mirror_point_y = lens_y + dy * 1.0
    
    # 미러에서 가장 가까운 점 찾기
    closest_idx = np.argmin(np.sqrt((x_mirror - mirror_point_x)**2 + (y_mirror - mirror_point_y)**2))
    mirror_x = x_mirror[closest_idx]
    mirror_y = y_mirror[closest_idx]
    
    plt.plot([lens_x, mirror_x], [lens_y, mirror_y], '-', color='darkred', alpha=0.8, linewidth=1)
    
    # 미러에서 스크린으로 반사
    # 입사각 = 반사각 원리 적용
    # 간단한 구현: 화면 상단에 도달하는 경로
    screen_y = screen_top_y - (angle - ray_angles_red[0]) / (ray_angles_red[-1] - ray_angles_red[0]) * screen_height * 0.6
    plt.plot([mirror_x, screen_x], [mirror_y, screen_y], '-', color='darkred', alpha=0.8, linewidth=1)

# 녹색 광선 (중앙)
ray_angles_green = [62, 75, 88]  # 도
for angle in ray_angles_green:
    # DMD에서 렌즈로
    plt.plot([dmd_x, lens_x], [dmd_y, dmd_y], '-', color='darkgreen', alpha=0.8, linewidth=1)
    
    # 렌즈에서 미러로 (각도 적용)
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    mirror_point_x = lens_x + dx * 1.0
    mirror_point_y = lens_y + dy * 1.0
    
    # 미러에서 가장 가까운 점 찾기
    closest_idx = np.argmin(np.sqrt((x_mirror - mirror_point_x)**2 + (y_mirror - mirror_point_y)**2))
    mirror_x = x_mirror[closest_idx]
    mirror_y = y_mirror[closest_idx]
    
    plt.plot([lens_x, mirror_x], [lens_y, mirror_y], '-', color='darkgreen', alpha=0.8, linewidth=1)
    
    # 미러에서 스크린으로 반사
    screen_y = screen_top_y - 0.3 - (angle - ray_angles_green[0]) / (ray_angles_green[-1] - ray_angles_green[0]) * screen_height * 0.6
    plt.plot([mirror_x, screen_x], [mirror_y, screen_y], '-', color='darkgreen', alpha=0.8, linewidth=1)

# 파란색 광선 (하단)
ray_angles_blue = [65, 78, 92]  # 도
for angle in ray_angles_blue:
    # DMD에서 렌즈로
    plt.plot([dmd_x, lens_x], [dmd_y, dmd_y], '-', color='orange', alpha=0.8, linewidth=1)
    
    # 렌즈에서 미러로 (각도 적용)
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    mirror_point_x = lens_x + dx * 1.0
    mirror_point_y = lens_y + dy * 1.0
    
    # 미러에서 가장 가까운 점 찾기
    closest_idx = np.argmin(np.sqrt((x_mirror - mirror_point_x)**2 + (y_mirror - mirror_point_y)**2))
    mirror_x = x_mirror[closest_idx]
    mirror_y = y_mirror[closest_idx]
    
    plt.plot([lens_x, mirror_x], [lens_y, mirror_y], '-', color='orange', alpha=0.8, linewidth=1)
    
    # 미러에서 스크린으로 반사
    screen_y = screen_top_y - 0.6 - (angle - ray_angles_blue[0]) / (ray_angles_blue[-1] - ray_angles_blue[0]) * screen_height * 0.4
    plt.plot([mirror_x, screen_x], [mirror_y, screen_y], '-', color='orange', alpha=0.8, linewidth=1)

# 축 및 레이아웃 설정
plt.axis('equal')
plt.grid(False)
plt.xlim(-1.5, 1.5)
plt.ylim(-0.8, 1.5)
plt.axis('off')  # 축 숨기기

plt.tight_layout()
plt.savefig('projector_geometry.png', dpi=300)
plt.show()

# 3차원 프로젝터 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 컴포넌트 위치 설정
dmd_pos = np.array([-1.0, 0, 0])
lens_pos = np.array([-0.5, 0, 0])
mirror_center = np.array([0.5, 0, 0])
mirror_y = 0  # 미러의 y 좌표

# 스크린 - X-Y 평면 참조용 스크린
screen_xy = np.array([-0.5, 0, 1.0])  # 참조용 스크린 중심
screen_size = 1.0

# X-Y 스크린 (참조용 스크린)
x_xy = np.linspace(screen_xy[0]-screen_size/2, screen_xy[0]+screen_size/2, 5)
y_xy = np.linspace(screen_xy[1]-screen_size/2, screen_xy[1]+screen_size/2, 5)
X_xy, Y_xy = np.meshgrid(x_xy, y_xy)
Z_xy = np.ones_like(X_xy) * screen_xy[2]
ax.plot_surface(X_xy, Y_xy, Z_xy, alpha=0.1, color='lightgray')  # 투명도 높게 설정

# Y-Z 평면 주 스크린 추가 (실제 스크린)
yz_screen_x = 1.5  # Y-Z 평면 스크린의 X 좌표
y_yz = np.linspace(-screen_size/2, screen_size/2, 5)
z_yz = np.linspace(0, screen_size, 5)
Y_yz, Z_yz = np.meshgrid(y_yz, z_yz)
X_yz = np.ones_like(Y_yz) * yz_screen_x
ax.plot_surface(X_yz, Y_yz, Z_yz, alpha=0.7, color='lightblue')  # 실제 스크린은 더 선명하게
ax.text(yz_screen_x, 0, screen_size/2, '실제 스크린 (Y-Z 평면)', fontsize=10)

# DMD 패널
dmd_width, dmd_height, dmd_depth = 0.4, 0.2, 0.05
x_dmd = np.array([dmd_pos[0]-dmd_width/2, dmd_pos[0]+dmd_width/2])
y_dmd = np.array([dmd_pos[1]-dmd_height/2, dmd_pos[1]+dmd_height/2])
z_dmd = np.array([dmd_pos[2], dmd_pos[2]+dmd_depth])
X_dmd, Y_dmd = np.meshgrid(x_dmd, y_dmd)
Z_dmd = np.ones_like(X_dmd) * z_dmd[0]
ax.plot_surface(X_dmd, Y_dmd, Z_dmd, color='lightblue', alpha=0.7)

# 렌즈 (단순화된 원통형)
lens_radius = 0.15
lens_thickness = 0.05
theta = np.linspace(0, 2*np.pi, 20)
z_lens = np.linspace(lens_pos[2]-lens_thickness/2, lens_pos[2]+lens_thickness/2, 2)
Theta, Z_lens = np.meshgrid(theta, z_lens)
X_lens = lens_pos[0] + lens_radius * np.cos(Theta)
Y_lens = lens_pos[1] + lens_radius * np.sin(Theta)
ax.plot_surface(X_lens, Y_lens, Z_lens, color='skyblue', alpha=0.5)

# 비구면 미러 (3D로 확장)
mirror_radius = 1.2
mirror_width = 0.8
theta_mirror = np.linspace(np.deg2rad(mirror_start_angle), np.deg2rad(mirror_end_angle), 20)
y_mirror_3d = np.linspace(-mirror_width/2, mirror_width/2, 10)
Theta_mirror, Y_mirror_3d = np.meshgrid(theta_mirror, y_mirror_3d)
X_mirror_3d = mirror_center[0] + mirror_radius * np.cos(Theta_mirror)
Z_mirror_3d = mirror_center[2] + mirror_radius * np.sin(Theta_mirror)
ax.plot_surface(X_mirror_3d, Y_mirror_3d, Z_mirror_3d, color='silver', alpha=0.7)

# 광선 경로 - 3D 공간에서 다른 색상으로
# 빨간색 광선
for angle in [60, 72, 84]:
    # DMD에서 렌즈로
    ax.plot([dmd_pos[0], lens_pos[0]], [dmd_pos[1], lens_pos[1]], [dmd_pos[2], lens_pos[2]], 
            '-', color='darkred', alpha=0.8, linewidth=1)
    
    # 렌즈에서 미러로 (3D 공간에서의 각도)
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dz = np.sin(angle_rad)
    
    # 미러 상의 점 찾기 (정확한 좌표 계산)
    mirror_x = mirror_center[0] + mirror_radius * np.cos(angle_rad+np.pi)
    mirror_z = mirror_center[2] + mirror_radius * np.sin(angle_rad+np.pi)
    
    ax.plot([lens_pos[0], mirror_x], [lens_pos[1], mirror_y], [lens_pos[2], mirror_z], 
            '-', color='darkred', alpha=0.8, linewidth=1)
    
    # 미러에서 Y-Z 평면 스크린으로 반사
    screen_z = screen_size/2 + (angle - 60) / (84 - 60) * screen_size * 0.4
    ax.plot([mirror_x, yz_screen_x], [mirror_y, (angle - 72) * 0.01], [mirror_z, screen_z], 
            '-', color='darkred', alpha=0.8, linewidth=1)

# 녹색 및 파란색 광선 (각각 다른 Y 평면에서)
for idx, (color, y_offset) in enumerate([('darkgreen', 0.1), ('orange', -0.1)]):
    for angle in [65, 75, 88]:
        # DMD에서 렌즈로
        ax.plot([dmd_pos[0], lens_pos[0]], [dmd_pos[1]+y_offset, lens_pos[1]+y_offset], 
                [dmd_pos[2], lens_pos[2]], '-', color=color, alpha=0.8, linewidth=1)
        
        # 렌즈에서 미러로
        angle_rad = np.deg2rad(angle)
        dx = np.cos(angle_rad)
        dz = np.sin(angle_rad)
        
        mirror_x = mirror_center[0] + mirror_radius * np.cos(angle_rad+np.pi)
        mirror_z = mirror_center[2] + mirror_radius * np.sin(angle_rad+np.pi)
        
        ax.plot([lens_pos[0], mirror_x], [lens_pos[1]+y_offset, lens_pos[1]+y_offset], 
                [lens_pos[2], mirror_z], '-', color=color, alpha=0.8, linewidth=1)
        
        # 미러에서 Y-Z 평면 스크린으로 반사
        screen_z = screen_size/2 + (angle - 65) / (88 - 65) * screen_size * 0.4
        ax.plot([mirror_x, yz_screen_x], [lens_pos[1]+y_offset, lens_pos[1]+y_offset + (angle - 75) * 0.01], 
                [mirror_z, screen_z], '-', color=color, alpha=0.8, linewidth=1)

# 컴포넌트 라벨
ax.text(dmd_pos[0], dmd_pos[1]-0.2, dmd_pos[2], 'DMD 패널', fontsize=10)
ax.text(lens_pos[0], lens_pos[1]-0.2, lens_pos[2], '렌즈', fontsize=10)
ax.text(mirror_center[0]+0.2, mirror_center[1], mirror_center[2], '비구면 미러', fontsize=10)
ax.text(screen_xy[0], screen_xy[1]-0.2, screen_xy[2]+0.5, '스크린', fontsize=10)

# 축 및 레이아웃 설정
ax.set_xlabel('X 축')
ax.set_ylabel('Y 축')
ax.set_zlabel('Z 축')
ax.set_title('프로젝터-스크린 3차원 기하학적 구성')
ax.view_init(elev=20, azim=-50)
ax.set_box_aspect([2, 1, 1])

plt.tight_layout()
plt.savefig('projector_three_screens_3d.png', dpi=300)
plt.show()

# X-Z 평면 뷰 (광선의 경로를 더 명확하게)
plt.figure(figsize=(10, 6))

# 컴포넌트 위치는 동일
# 스크린 그리기 (수직선)
plt.plot([screen_x, screen_x], [0, screen_top_y], 'k-', linewidth=2)
plt.text(screen_x-0.15, screen_top_y+0.05, '스크린', fontsize=12)

# DMD 패널 그리기 (X-Z 평면에서는 선으로)
plt.plot([dmd_x-dmd_width/2, dmd_x+dmd_width/2], [dmd_y, dmd_y], 'k-', linewidth=2)
plt.text(dmd_x-dmd_width/2-0.25, dmd_y, 'DMD 패널', fontsize=12)

# 렌즈 그리기 (X-Z 평면에서는 세로선으로)
plt.plot([lens_x, lens_x], [dmd_y-lens_radius, dmd_y+lens_radius], 'b-', linewidth=3)
plt.text(lens_x, dmd_y-lens_radius-0.1, '렌즈', fontsize=12)

# 비구면 미러 그리기
plt.plot(x_mirror, y_mirror, '-', linewidth=3, color='gray')
plt.text(max(x_mirror)+0.05, y_mirror[0], '비구면 미러 표면', fontsize=12)

# 광선 경로 - X-Z 평면에서 보기
# 세 색상의 광선
for i, (color, angles) in enumerate([('darkred', ray_angles_red), 
                                     ('darkgreen', ray_angles_green), 
                                     ('orange', ray_angles_blue)]):
    for j, angle in enumerate(angles):
        # DMD에서 렌즈로
        plt.plot([dmd_x, lens_x], [dmd_y, dmd_y], '-', color=color, alpha=0.8, linewidth=1)
        
        # 렌즈에서 미러로
        angle_rad = np.deg2rad(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        mirror_point_x = lens_x + dx * 1.0
        mirror_point_y = lens_y + dy * 1.0
        
        # 미러에서 가장 가까운 점 찾기
        closest_idx = np.argmin(np.sqrt((x_mirror - mirror_point_x)**2 + (y_mirror - mirror_point_y)**2))
        mirror_x = x_mirror[closest_idx]
        mirror_y = y_mirror[closest_idx]
        
        plt.plot([lens_x, mirror_x], [lens_y, mirror_y], '-', color=color, alpha=0.8, linewidth=1)
        
        # 미러에서 스크린으로 반사
        screen_z = screen_top_y - (i*0.3 + j*0.1)  # 각 색상별로 스크린의 다른 위치에 도달
        plt.plot([mirror_x, screen_xy[0]], [mirror_y, screen_z], '-', color=color, alpha=0.8, linewidth=1)

# 설명 텍스트 추가
plt.text(0, -0.5, '측면도 (X-Z 평면): 이 뷰에서 광선 경로를 명확하게 확인할 수 있습니다.\n'
                  '빨간색, 녹색, 주황색 광선이 렌즈를 통과한 후 비구면 미러에서 반사되어\n'
                  '스크린의 서로 다른 위치에 도달하는 것을 보여줍니다.', fontsize=10)

# 축 및 레이아웃 설정
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.xlim(-1.5, 1.5)
plt.ylim(-0.8, 1.5)

plt.tight_layout()
plt.savefig('projector_xz_plane.png', dpi=300)
plt.show()

# Stokes 파라미터 계산 및 시각화 함수 추가
def calc_stokes(jones):
    """
    Jones 벡터로부터 Stokes 파라미터를 계산하는 함수
    
    파라미터:
    - jones: 2차원 Jones 벡터 (복소수)
    
    반환값:
    - [S0, S1, S2, S3]: Stokes 파라미터 (4차원 실수 벡터)
      - S0: 총 강도
      - S1: 수평/수직 선형 편광의 차이
      - S2: +45°/-45° 선형 편광의 차이
      - S3: 우원형/좌원형 편광의 차이
    """
    Ex, Ey = jones[0], jones[1]
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = 2 * np.imag(Ex * np.conj(Ey))
    return np.array([S0, S1, S2, S3])

def plot_poincare_sphere(stokes_list, colors, labels=None, title=None):
    """
    푸앵카레 구면에 편광 상태를 표시합니다.
    """
    # 주석 처리된 푸앵카레 구면 시각화
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 구면 그리기
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='gray')
    
    # 좌표축 그리기
    ax.quiver(-1.5, 0, 0, 3, 0, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, -1.5, 0, 0, 3, 0, color='k', arrow_length_ratio=0.1)
    ax.quiver(0, 0, -1.5, 0, 0, 3, color='k', arrow_length_ratio=0.1)
    
    # 편광 상태 표시
    for i, stokes in enumerate(stokes_list):
        if np.any(np.isnan(stokes)):
            continue
        color = colors[i] if isinstance(colors, list) else colors
        label = labels[i] if labels is not None else f'State {i+1}'
        ax.scatter(stokes[1], stokes[2], stokes[3], c=color, s=100, label=label)
    
    # 축 레이블
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    
    if title:
        ax.set_title(title)
    
    ax.legend()
    plt.tight_layout()
    """
    pass

def plot_stokes_comparison(stokes_before, stokes_after, labels, title=None):
    """
    Stokes 파라미터 변화를 막대 그래프로 비교 시각화하는 함수
    
    파라미터:
    - stokes_before: 처리 전 Stokes 파라미터 리스트
    - stokes_after: 처리 후 Stokes 파라미터 리스트
    - labels: 각 Stokes 벡터에 대한 레이블 리스트
    - title: 그래프 제목 (기본값=None)
    
    반환값:
    - fig: 그래프 객체
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 정규화된 Stokes 파라미터 계산
    S1_before = [s[1]/s[0] for s in stokes_before]
    S2_before = [s[2]/s[0] for s in stokes_before]
    S3_before = [s[3]/s[0] for s in stokes_before]
    
    S1_after = [s[1]/s[0] for s in stokes_after]
    S2_after = [s[2]/s[0] for s in stokes_after]
    S3_after = [s[3]/s[0] for s in stokes_after]
    
    # x 위치 설정
    x = np.arange(len(labels))
    width = 0.35
    
    # S1 (수평/수직) 그래프
    axes[0].bar(x - width/2, S1_before, width, label='처리 전')
    axes[0].bar(x + width/2, S1_after, width, label='처리 후')
    axes[0].set_ylabel('S1/S0 (수평/수직)')
    axes[0].set_title('수평(+1)/수직(-1) 편광 성분 변화')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim([-1.1, 1.1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # S2 (+45/-45) 그래프
    axes[1].bar(x - width/2, S2_before, width, label='처리 전')
    axes[1].bar(x + width/2, S2_after, width, label='처리 후')
    axes[1].set_ylabel('S2/S0 (+45°/-45°)')
    axes[1].set_title('+45°(+1)/-45°(-1) 편광 성분 변화')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim([-1.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # S3 (우/좌 원형) 그래프
    axes[2].bar(x - width/2, S3_before, width, label='처리 전')
    axes[2].bar(x + width/2, S3_after, width, label='처리 후')
    axes[2].set_ylabel('S3/S0 (우/좌 원형)')
    axes[2].set_title('우원형(+1)/좌원형(-1) 편광 성분 변화')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim([-1.1, 1.1])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Stokes 파라미터 변화 비교', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

# 광선 추적을 위한 함수 추가

class PolarizationAnalyzer:
    """
    편광 상태 분석 및 시뮬레이션을 위한 클래스
    """
    def __init__(self):
        # 기본 존스 벡터 정의
        self.H = np.array([1, 0])  # 수평 편광 (p-편광)
        self.V = np.array([0, 1])  # 수직 편광 (s-편광)
        
        # 색상 정의 (RGB)
        self.colors = {
            'red': {'wavelength': 650e-9, 'color': 'red'},
            'green': {'wavelength': 520e-9, 'color': 'green'},
            'blue': {'wavelength': 450e-9, 'color': 'blue'}
        }
        
        # 알루미늄의 파장별 굴절률 (실제 데이터에 맞게 수정 필요)
        # 형식: 파장(nm): [n, k] (복소 굴절률)
        self.al_refractive_index = {
            450: [0.61, 5.92],  # 블루
            520: [0.83, 6.04],  # 그린 
            650: [1.24, 7.21]   # 레드
        }
    
    def fresnel_coeffs(self, n1, n2, theta_i, wavelength_nm):
        """
        프레넬 반사 계수 계산 (s-편광, p-편광)
        :param n1: 입사 매질의 굴절률
        :param n2: 투과 매질의 복소 굴절률 (n + ik)
        :param theta_i: 입사각 (라디안)
        :param wavelength_nm: 파장 (nm)
        :return: (rs, rp) - s-편광과 p-편광의 반사 계수 (복소수)
        """
        # 파장에 따른 알루미늄 복소 굴절률 가져오기
        n2_complex = complex(n2[0], n2[1])
        
        # 스넬의 법칙을 사용하여 투과각 계산 (복소수)
        n1_sin_theta_i = n1 * np.sin(theta_i)
        theta_t = np.arcsin(n1_sin_theta_i / n2_complex)
        
        # s-편광 반사 계수 (TE 모드)
        rs = (n1 * np.cos(theta_i) - n2_complex * np.cos(theta_t)) / \
             (n1 * np.cos(theta_i) + n2_complex * np.cos(theta_t))
        
        # p-편광 반사 계수 (TM 모드)
        rp = (n2_complex * np.cos(theta_i) - n1 * np.cos(theta_t)) / \
             (n2_complex * np.cos(theta_i) + n1 * np.cos(theta_t))
        
        return rs, rp
    
    def propagate(self, initial_state, theta_deg, wavelength_nm, apply_hwp=False, selective_hwp=False):
        """
        편광 상태를 여러 광학 요소를 통해 전파
        :param initial_state: 초기 편광 상태 (존스 벡터)
        :param theta_deg: 입사각 (도)
        :param wavelength_nm: 파장 (nm)
        :param apply_hwp: 반파장판 적용 여부
        :param selective_hwp: RGB 중 GB에만 반파장판 적용 여부
        :return: 최종 편광 상태 (존스 벡터) 및 강도
        """
        # 도를 라디안으로 변환
        theta_rad = np.radians(theta_deg)
        
        # 가장 가까운 파장 키 찾기
        wavelength_key = min(self.al_refractive_index.keys(), 
                           key=lambda k: abs(k - wavelength_nm))
        
        # 알루미늄 반사면에 대한 프레넬 계수 계산
        rs, rp = self.fresnel_coeffs(1.0, self.al_refractive_index[wavelength_key], 
                                   theta_rad, wavelength_nm)
        
        # 현재 상태 복사
        current_state = initial_state.copy()
        
        # 선택적 반파장판 적용 (GB만 처리)
        if selective_hwp and wavelength_nm < 600e-9:  # GB에만 적용
            # 반파장판 존스 행렬 (고속축 수평, π 위상 지연)
            hwp = np.array([[1, 0], [0, -1]])
            current_state = np.dot(hwp, current_state)
        
        # 일반 반파장판 적용
        elif apply_hwp:
            # 반파장판 존스 행렬 (고속축 수평, π 위상 지연)
            hwp = np.array([[1, 0], [0, -1]])
            current_state = np.dot(hwp, current_state)
        
        # 알루미늄 반사면에서 반사
        # 존스 행렬로 변환하여 적용
        reflection_matrix = np.array([[rp, 0], [0, rs]])
        reflected_state = np.dot(reflection_matrix, current_state)
        
        # 반사 후 강도 계산
        intensity = np.abs(reflected_state[0])**2 + np.abs(reflected_state[1])**2
        
        return reflected_state, intensity
    
    def calc_stokes(self, jones_vector):
        """
        존스 벡터로부터 스토크스 파라미터 계산
        :param jones_vector: 복소 존스 벡터 [Ex, Ey]
        :return: 스토크스 벡터 [S0, S1, S2, S3]
        """
        Ex = jones_vector[0]
        Ey = jones_vector[1]
        
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1 = np.abs(Ex)**2 - np.abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = 2 * np.imag(Ex * np.conj(Ey))
        
        return np.array([S0, S1, S2, S3])
    
    def analyze_angle_range(self, angles_deg, apply_hwp=False, selective_hwp=False):
        """
        주어진 각도 범위에서 편광 상태 및 강도 분석
        :param angles_deg: 분석할 각도 범위 (도)
        :param apply_hwp: 반파장판 적용 여부
        :param selective_hwp: RGB 중 GB에만 반파장판 적용 여부
        :return: 각도별, 색상별 결과 딕셔너리
        """
        results = {"angles": angles_deg}
        
        for color_name, color_info in self.colors.items():
            wavelength = color_info['wavelength']
            
            # 초기 상태 (수평 편광)
            initial_state = self.H.copy()
            
            intensity_list = []
            stokes_list = []
            
            for angle in angles_deg:
                # 주어진 각도에서 빛 전파
                final_state, intensity = self.propagate(
                    initial_state, angle, wavelength * 1e9, 
                    apply_hwp=apply_hwp, selective_hwp=selective_hwp
                )
                
                # 스토크스 파라미터 계산
                stokes = self.calc_stokes(final_state)
                
                intensity_list.append(intensity)
                stokes_list.append(stokes)
            
            results[color_name] = {
                "intensity": intensity_list,
                "stokes": stokes_list
            }
        
        return results
    
    def plot_analysis_results(self, results, title=None):
        """
        분석 결과 시각화
        :param results: analyze_angle_range 메서드의 결과
        :param title: 그래프 제목
        :return: 그래프 객체
        """
        angles = results["angles"]
        
        # 강도 그래프
        plt.figure(figsize=(12, 6))
        
        for color_name, color_info in self.colors.items():
            intensity = results[color_name]["intensity"]
            plt.plot(angles, intensity, label=color_name.capitalize(), 
                   color=color_info['color'], linewidth=2)
        
        plt.xlabel('입사각 (도)')
        plt.ylabel('반사 강도')
        plt.title('각도별 반사 강도' if title is None else title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 푸앵카레 구면 시각화 (특정 각도에서의 편광 상태)
        stokes_list = []
        labels = []
        colors = []
        
        for color_name, color_info in self.colors.items():
            # 중간 각도의 스토크스 파라미터 선택
            mid_idx = len(angles) // 2
            stokes = results[color_name]["stokes"][mid_idx]
            stokes_list.append(stokes)
            labels.append(f"{color_name.capitalize()} ({angles[mid_idx]}°)")
            colors.append(color_info['color'])
        
        # 푸앵카레 구면 표시
        fig_poincare = plot_poincare_sphere(stokes_list, colors, labels, 
                                         title=f'푸앵카레 구면 상의 편광 상태 ({angles[mid_idx]}°)')
        
        return fig_poincare

def run_polarization_analysis_example():
    """
    PolarizationAnalyzer 클래스 사용 예시
    """
    analyzer = PolarizationAnalyzer()
    
    # 10도에서 80도까지 10도 간격으로 분석
    angles = np.arange(10, 81, 10)
    
    # 일반 반사 분석
    results_normal = analyzer.analyze_angle_range(angles)
    
    # 반파장판 적용 분석
    results_hwp = analyzer.analyze_angle_range(angles, apply_hwp=True)
    
    # 선택적(GB만) 반파장판 적용 분석
    results_selective = analyzer.analyze_angle_range(angles, selective_hwp=True)
    
    # 결과 출력
    for angle_idx, angle in enumerate(angles):
        print(f"\n각도: {angle}°")
        print("-" * 40)
        
        print("일반 반사:")
        for color in ["red", "green", "blue"]:
            intensity = results_normal[color]["intensity"][angle_idx]
            print(f"  {color.capitalize()}: {intensity:.3f}")
        
        print("\n반파장판 적용 후:")
        for color in ["red", "green", "blue"]:
            intensity = results_hwp[color]["intensity"][angle_idx]
            print(f"  {color.capitalize()}: {intensity:.3f}")
        
        print("\n선택적(GB만) 반파장판 적용 후:")
        for color in ["red", "green", "blue"]:
            intensity = results_selective[color]["intensity"][angle_idx]
            print(f"  {color.capitalize()}: {intensity:.3f}")
    
    # 그래프 시각화
    analyzer.plot_analysis_results(results_normal, "일반 반사")
    analyzer.plot_analysis_results(results_hwp, "반파장판 적용 후")
    analyzer.plot_analysis_results(results_selective, "선택적(GB만) 반파장판 적용 후")
    
    # 푸앵카레 구면 시각화
    # 이미 plot_analysis_results에서 구현됨
    
    plt.show()

# 광선 추적을 위한 함수 추가
try:
    from raysect.optical import World, Material, Sphere, Mesh, Ray
    from raysect.optical.material import Lambert, Checkerboard, Conductor
    from raysect.optical.material.dielectric import Dielectric
    # roughconductor 모듈이 아닌 conductor 모듈에서 RoughConductor 가져오기
    from raysect.optical.material.conductor import RoughConductor
    from raysect.primitive import Box, Cylinder
    from raysect.core import Point3D, Vector3D, translate, rotate, AffineMatrix3D
    from raysect.optical.observer import PinholeCamera
    from raysect.optical.observer.imaging import RGBPipeline2D
    from raysect.optical.library.spectra import rgb
    from raysect.optical.library.metal import Aluminium, Silver
    # Light 클래스가 필요하다면 다음과 같이 가져옵니다
    from raysect.optical.material.debug import Light
    # PointLight 경로를 정확히 확인할 필요가 있습니다
    # 가능한 대안:
    from raysect.optical.material import UniformSurfaceEmitter, AnisotropicSurfaceEmitter
    # 또는 
    from raysect.optical.material import UniformSurfaceEmitter
    from raysect.optical.material import AnisotropicSurfaceEmitter
    # Raysect imports
    from raysect.optical import World, translate, rotate, Point3D, d65_white
    from raysect.optical.observer import PinholeCamera
    from raysect.optical.material import UniformSurfaceEmitter
    from raysect.optical.library import RoughAluminium
    from raysect.primitive import Box
    from raysect.optical.material import AnisotropicSurfaceEmitter

    RAYSECT_AVAILABLE = True
except ImportError as e:
    RAYSECT_AVAILABLE = False
    print(f"Raysect 라이브러리를 찾을 수 없습니다. 오류: {e}")
    print("광선 추적 기능은 비활성화됩니다.")

def render_scene(camera_positions, output_prefix="render", resolution=(800, 600)):
    """
    Raysect를 사용하여 장면을 렌더링합니다.
    :param camera_positions: 카메라 위치 목록 [(x, y, z, target_x, target_y, target_z), ...]
    :param output_prefix: 출력 파일 접두사
    :param resolution: 렌더링 해상도
    """
    if not RAYSECT_AVAILABLE:
        print("Raysect 라이브러리가 설치되지 않아 렌더링을 수행할 수 없습니다.")
        return
    
    # 월드 생성
    world = World()
    
    # 물체 위치 정의
    dmd_position = Point3D(0, 0, 0)
    lens_position = Point3D(0, 0, 10)
    mirror_position = Point3D(5, 0, 15)
    screen_distance = 30
    
    # DMD 패널 생성 (복잡한 반사 특성을 가진 거친 도체)
    dmd = Box(Point3D(-1, -1, -0.1), Point3D(1, 1, 0.1))
    dmd.material = RoughConductor(Aluminium(), 0.2)
    dmd.parent = world
    dmd.transform = rotate(45, 0, 0) * translate(dmd_position.x, dmd_position.y, dmd_position.z)
    
    # 렌즈 생성 (유전체 재질)
    lens = Cylinder(1.5, 0.5)
    lens.material = Dielectric(1.5)  # 굴절률 1.5의 유리
    lens.parent = world
    lens.transform = rotate(0, 90, 0) * translate(lens_position.x, lens_position.y, lens_position.z)
    
    # 비구면 거울 생성 (알루미늄 거친 도체)
    mirror = Box(Point3D(-2, -2, -0.1), Point3D(2, 2, 0.1))
    mirror.material = RoughConductor(Aluminium(), 0.05)  # 매우 매끄러운 알루미늄
    mirror.parent = world
    mirror.transform = rotate(30, 0, 0) * translate(mirror_position.x, mirror_position.y, mirror_position.z)
    
    # 스크린 생성
    screen = Box(Point3D(-15, -15, -0.1), Point3D(15, 15, 0.1))
    screen.material = Lambert(rgb(0.8, 0.8, 0.8))
    screen.parent = world
    screen.transform = translate(10, 0, screen_distance)
    
    # 바닥 생성
    floor = Box(Point3D(-50, -0.5, -50), Point3D(50, 0, 50))
    floor.material = Checkerboard(4, Lambert(rgb(0.8, 0.8, 0.8)), Lambert(rgb(0.2, 0.2, 0.2)))
    floor.parent = world
    
    # DMD에서 나오는 빛 (RGB 포인트 광원)
    # DMD에서 나오는 빛 (RGB 광원)
    red_sphere = Sphere(0.2, transform=translate(dmd_position.x-0.5, dmd_position.y+0.5, dmd_position.z+0.2))
    red_sphere.material = UniformSurfaceEmitter(rgb(1.0, 0.0, 0.0), 50)
    red_sphere.parent = world

    green_sphere = Sphere(0.2, transform=translate(dmd_position.x, dmd_position.y+0.5, dmd_position.z+0.2))
    green_sphere.material = UniformSurfaceEmitter(rgb(0.0, 1.0, 0.0), 50)
    green_sphere.parent = world

    blue_sphere = Sphere(0.2, transform=translate(dmd_position.x+0.5, dmd_position.y+0.5, dmd_position.z+0.2))
    blue_sphere.material = UniformSurfaceEmitter(rgb(0.0, 0.0, 1.0), 50)
    blue_sphere.parent = world

    # 추가 앰비언트 라이트 (약한 흰색 조명)
    ambient_sphere = Sphere(0.3, transform=translate(0, 10, 0))
    ambient_sphere.material = UniformSurfaceEmitter(rgb(0.2, 0.2, 0.2), 100)
    ambient_sphere.parent = world
    
    # 추가 앰비언트 라이트 (약한 흰색 조명)
    # 추가 앰비언트 라이트 (약한 흰색 조명)
    ambient_sphere = Sphere(0.3, transform=translate(0, 10, 0))
    ambient_sphere.material = UniformSurfaceEmitter(rgb(0.2, 0.2, 0.2), 100)
    ambient_sphere.parent = world
    
    # 각 카메라 위치에서 렌더링
    for i, (x, y, z, target_x, target_y, target_z) in enumerate(camera_positions):
        # 카메라 설정
        camera = PinholeCamera(
            parent=world,
            position=Point3D(x, y, z),
            look_at=Point3D(target_x, target_y, target_z),
            resolution=resolution,
            fov=45
        )
        
        # 렌더링 파이프라인 설정
        pipeline = RGBPipeline2D()
        camera.pipelines = [pipeline]
        camera.pixel_samples = 10  # 샘플 수 (높을수록 품질 좋아지지만 시간 오래 걸림)
        
        # 렌더링 실행
        print(f"카메라 위치 {i+1}에서 렌더링 중...")
        camera.observe()
        
        # 결과 저장
        pipeline.save(f"{output_prefix}_camera{i+1}.png")
        print(f"이미지가 {output_prefix}_camera{i+1}.png 로 저장되었습니다.")
    
    return world

def visualize_ray_paths(num_rays=20, output_file="ray_paths.png"):
    """
    DMD에서 스크린까지의 광선 경로를 시각화합니다.
    :param num_rays: 시각화할 광선 수
    :param output_file: 출력 파일 이름
    """
    if not RAYSECT_AVAILABLE:
        print("Raysect 라이브러리가 설치되지 않아 광선 경로를 시각화할 수 없습니다.")
        return
    
    # 물체 위치 정의
    dmd_position = Point3D(0, 0, 0)
    lens_position = Point3D(0, 0, 10)
    mirror_position = Point3D(5, 0, 15)
    screen_distance = 30
    
    # 3D 그래프 설정
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # DMD 그리기 - 0도 (회전 없음)
    dmd_x, dmd_y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    dmd_z = np.zeros_like(dmd_x)
    # DMD는 0도 (회전 없음)
    ax.plot_surface(dmd_x + dmd_position.x, dmd_y + dmd_position.y, 
                   dmd_z + dmd_position.z, alpha=0.3, color='blue')
    
    # 렌즈 그리기
    lens_r, lens_theta = np.meshgrid(np.linspace(0, 1.5, 10), np.linspace(0, 2*np.pi, 20))
    lens_x = lens_r * np.cos(lens_theta)
    lens_y = lens_r * np.sin(lens_theta)
    lens_z = np.zeros_like(lens_x)
    ax.plot_surface(lens_x + lens_position.x, lens_y + lens_position.y, 
                  lens_z + lens_position.z, alpha=0.3, color='cyan')
    
    # 비구면 거울 그리기 - 각도를 -30도로 변경
    mirror_x, mirror_y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    mirror_z = np.zeros_like(mirror_x)
    # -30도 회전 적용 (부호 변경)
    mirror_z_rot = mirror_z - mirror_x * np.sin(np.radians(30))  # 부호 변경
    mirror_x_rot = mirror_x * np.cos(np.radians(30))
    ax.plot_surface(mirror_x_rot + mirror_position.x, mirror_y + mirror_position.y, 
                  mirror_z_rot + mirror_position.z, alpha=0.3, color='silver')
    
    # 스크린 그리기
    screen_x, screen_y = np.meshgrid(np.linspace(-15, 15, 10), np.linspace(-15, 15, 10))
    screen_z = np.zeros_like(screen_x)
    ax.plot_surface(screen_x + 10, screen_y, screen_z + screen_distance, alpha=0.3, color='lightgray')
    
    # DMD에서 출발하는 광선 시각화
    for i in range(num_rays):
        # 광선 시작점 (DMD 표면의 무작위 위치)
        start_x = np.random.uniform(-0.8, 0.8)
        start_y = np.random.uniform(-0.8, 0.8)
        # 45도 회전 적용
        start_x_rot = start_x
        start_z_rot = -start_x
        
        # 렌즈 방향으로의 광선
        direction_to_lens = np.array([lens_position.x, lens_position.y, lens_position.z]) - \
                           np.array([start_x_rot, start_y, start_z_rot + dmd_position.z])
        direction_to_lens = direction_to_lens / np.linalg.norm(direction_to_lens)
        
        # 렌즈 도달 지점 계산
        lens_hit = np.array([start_x_rot, start_y, start_z_rot + dmd_position.z]) + \
                  direction_to_lens * 10  # 거리 10 가정
        
        # 거울 방향으로의 광선
        direction_to_mirror = np.array([mirror_position.x, mirror_position.y, mirror_position.z]) - \
                             lens_hit
        direction_to_mirror = direction_to_mirror / np.linalg.norm(direction_to_mirror)
        
        # 거울 도달 지점 계산
        mirror_hit = lens_hit + direction_to_mirror * 8  # 거리 8 가정
        
        # 거울의 법선 벡터 계산 (30도 기울어진 거울)
        mirror_normal = np.array([np.sin(np.radians(90)), 0, np.cos(np.radians(90))])
        mirror_normal = mirror_normal / np.linalg.norm(mirror_normal)
        
        # 물리적 반사 법칙을 사용하여 반사 방향 계산: r = d - 2(d·n)n
        # 여기서 d는 입사 방향, n은 법선 벡터, r은 반사 방향
        reflection = direction_to_mirror - 2 * np.dot(direction_to_mirror, mirror_normal) * mirror_normal
        reflection = reflection / np.linalg.norm(reflection)
        
        # 스크린 도달 지점 계산 (반사된 방향으로 진행)
        screen_distance_from_mirror = np.abs((screen_distance - mirror_hit[2]) / reflection[2])
        screen_hit = mirror_hit + reflection * screen_distance_from_mirror
        
        # 광선 경로 그리기 (DMD -> 렌즈 -> 거울 -> 스크린)
        color = np.random.choice(['r', 'g', 'b'])
        ax.plot([start_x_rot, lens_hit[0]], 
               [start_y, lens_hit[1]], 
               [start_z_rot + dmd_position.z, lens_hit[2]], 
               color, alpha=0.7)
        ax.plot([lens_hit[0], mirror_hit[0]], 
               [lens_hit[1], mirror_hit[1]], 
               [lens_hit[2], mirror_hit[2]], 
               color, alpha=0.7)
        ax.plot([mirror_hit[0], screen_hit[0]], 
               [mirror_hit[1], screen_hit[1]], 
               [mirror_hit[2], screen_hit[2]], 
               color, alpha=0.7)
    
    # 그래프 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('DMD에서 스크린까지의 광선 경로')
    
    # 주요 컴포넌트 위치 표시
    ax.text(dmd_position.x, dmd_position.y, dmd_position.z, 'DMD', color='blue')
    ax.text(lens_position.x, lens_position.y, lens_position.z, 'Lens', color='cyan')
    ax.text(mirror_position.x, mirror_position.y, mirror_position.z, 'Mirror', color='silver')
    ax.text(10, 0, screen_distance, 'Screen', color='black')
    
    # 보기 각도 설정
    ax.view_init(elev=20, azim=-35)
    
    # 저장
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"광선 경로 시각화가 {output_file}에 저장되었습니다.")
    
    return fig

# Raysect 없이 광선 추적 시뮬레이션 (대안)
def render_scene_alternative(camera_positions, output_prefix="render", resolution=(800, 600)):
    """
    Matplotlib를 사용하여 장면을 렌더링합니다. (Raysect 대체용)
    :param camera_positions: 카메라 위치 목록 [(x, y, z, target_x, target_y, target_z), ...]
    :param output_prefix: 출력 파일 접두사
    :param resolution: 렌더링 해상도
    """
    # 물체 위치 정의
    dmd_position = np.array([0, 0, 0])
    lens_position = np.array([0, 0, 10])
    mirror_position = np.array([5, 0, 15])
    screen_distance = 30
    
    for i, (x, y, z, target_x, target_y, target_z) in enumerate(camera_positions):
        # 3D 그래프 설정
        fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # DMD 패널 그리기
        dmd_x, dmd_y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        dmd_z = np.zeros_like(dmd_x)
        # 45도 회전 적용
        dmd_x_rot = dmd_x.copy()
        dmd_z_rot = dmd_z - dmd_x
        ax.plot_surface(dmd_x_rot, dmd_y, dmd_z_rot + dmd_position[2], alpha=0.3, color='blue')
        
        # 렌즈 그리기
        lens_r, lens_theta = np.meshgrid(np.linspace(0, 1.5, 10), np.linspace(0, 2*np.pi, 20))
        lens_x = lens_r * np.cos(lens_theta)
        lens_y = lens_r * np.sin(lens_theta)
        lens_z = np.zeros_like(lens_x)
        ax.plot_surface(lens_x + lens_position[0], lens_y + lens_position[1], 
                      lens_z + lens_position[2], alpha=0.3, color='cyan')
        
        # 비구면 거울 그리기
        mirror_x, mirror_y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
        mirror_z = np.zeros_like(mirror_x)
        # 30도 회전 적용
        mirror_z_rot = mirror_z + mirror_x * np.sin(np.radians(30))
        mirror_x_rot = mirror_x * np.cos(np.radians(30))
        ax.plot_surface(mirror_x_rot + mirror_position[0], mirror_y + mirror_position[1], 
                      mirror_z_rot + mirror_position[2], alpha=0.3, color='silver')
        
        # 스크린 그리기
        screen_x, screen_y = np.meshgrid(np.linspace(-15, 15, 10), np.linspace(-15, 15, 10))
        screen_z = np.zeros_like(screen_x)
        ax.plot_surface(screen_x + 10, screen_y, screen_z + screen_distance, alpha=0.3, color='lightgray')
        
        # 광선 시각화
        for j in range(20):  # 20개의 광선
            # 광선 시작점 (DMD 표면의 무작위 위치)
            start_x = np.random.uniform(-0.8, 0.8)
            start_y = np.random.uniform(-0.8, 0.8)
            # 45도 회전 적용
            start_x_rot = start_x
            start_z_rot = -start_x
            
            # 렌즈 방향으로의 광선
            direction_to_lens = lens_position - np.array([start_x_rot, start_y, start_z_rot + dmd_position[2]])
            direction_to_lens = direction_to_lens / np.linalg.norm(direction_to_lens)
            
            # 렌즈 도달 지점 계산
            lens_hit = np.array([start_x_rot, start_y, start_z_rot + dmd_position[2]]) + direction_to_lens * 10
            
            # 거울 방향으로의 광선
            direction_to_mirror = mirror_position - lens_hit
            direction_to_mirror = direction_to_mirror / np.linalg.norm(direction_to_mirror)
            
            # 거울 도달 지점 계산
            mirror_hit = lens_hit + direction_to_mirror * 8
            
            # 스크린 방향으로의 광선 (반사 각도 단순화)
            direction_to_screen = np.array([10, 0, screen_distance]) - mirror_hit
            direction_to_screen = direction_to_screen / np.linalg.norm(direction_to_screen)
            
            # 스크린 도달 지점 계산
            screen_hit = mirror_hit + direction_to_screen * 20
            
            # 광선 색상 (RGB 중 하나)
            color = np.random.choice(['r', 'g', 'b'])
            
            # 광선 경로 그리기
            ax.plot([start_x_rot, lens_hit[0]], [start_y, lens_hit[1]], 
                    [start_z_rot + dmd_position[2], lens_hit[2]], color, alpha=0.7)
            ax.plot([lens_hit[0], mirror_hit[0]], [lens_hit[1], mirror_hit[1]], 
                    [lens_hit[2], mirror_hit[2]], color, alpha=0.7)
            ax.plot([mirror_hit[0], screen_hit[0]], [mirror_hit[1], screen_hit[1]], 
                    [mirror_hit[2], screen_hit[2]], color, alpha=0.7)
        
        # 그래프 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('초단초점 프로젝터 광학계 시뮬레이션')
        
        # 주요 컴포넌트 위치 표시
        ax.text(dmd_position[0], dmd_position[1], dmd_position[2], 'DMD', color='blue')
        ax.text(lens_position[0], lens_position[1], lens_position[2], 'Lens', color='cyan')
        ax.text(mirror_position[0], mirror_position[1], mirror_position[2], 'Mirror', color='silver')
        ax.text(10, 0, screen_distance, 'Screen', color='black')
        
        # 보기 각도 설정
        ax.view_init(elev=20, azim=-35)
        
        # 카메라 위치 설정
        camera_pos = np.array([x, y, z])
        target_pos = np.array([target_x, target_y, target_z])
        
        # 카메라 위치 표시
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], color='red', s=100, label='Camera')
        ax.plot([camera_pos[0], target_pos[0]], [camera_pos[1], target_pos[1]], 
               [camera_pos[2], target_pos[2]], 'r--', alpha=0.5)
        
        # 저장
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_camera{i+1}.png", dpi=300)
        print(f"이미지가 {output_prefix}_camera{i+1}.png 로 저장되었습니다.")
        plt.close()
    
    return

def visualize_ray_paths_alternative(num_rays=20, output_file="ray_paths.png"):
    """
    DMD에서 스크린까지의 광선 경로를 시각화합니다. (Raysect 없이)
    :param num_rays: 시각화할 광선 수
    :param output_file: 출력 파일 이름
    """
    # 물체 위치 정의
    dmd_position = np.array([0, 0, 0])
    lens_position = np.array([0, 0, 10])
    mirror_position = np.array([5, 0, 15])
    screen_distance = 30
    screen_offset_x = -5  # 스크린 X축 오프셋
    screen_offset_y = 2   # 스크린 Y축 오프셋
    
    # 3D 그래프 설정
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # DMD 그리기 - 0도 (회전 없음)
    dmd_x, dmd_y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    dmd_z = np.zeros_like(dmd_x)
    ax.plot_surface(dmd_x + dmd_position[0], dmd_y + dmd_position[1], 
                   dmd_z + dmd_position[2], alpha=0.3, color='blue')
    
    # 렌즈 그리기
    lens_r, lens_theta = np.meshgrid(np.linspace(0, 1.5, 10), np.linspace(0, 2*np.pi, 20))
    lens_x = lens_r * np.cos(lens_theta)
    lens_y = lens_r * np.sin(lens_theta)
    lens_z = np.zeros_like(lens_x)
    ax.plot_surface(lens_x + lens_position[0], lens_y + lens_position[1], 
                  lens_z + lens_position[2], alpha=0.3, color='cyan')
    
    # 비구면 거울 그리기 - 각도를 -30도로 변경
    mirror_x, mirror_y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    mirror_z = np.zeros_like(mirror_x)
    mirror_z_rot = mirror_z - mirror_x * np.sin(np.radians(30))
    mirror_x_rot = mirror_x * np.cos(np.radians(30))
    ax.plot_surface(mirror_x_rot + mirror_position[0], mirror_y + mirror_position[1], 
                  mirror_z_rot + mirror_position[2], alpha=0.3, color='silver')
    
    # Y-Z 평면 스크린 그리기 (오프셋 적용)
    screen_y, screen_z = np.meshgrid(
        np.linspace(-10 + screen_offset_y, 10 + screen_offset_y, 10),
        np.linspace(0, screen_distance, 10)
    )
    screen_x = np.full_like(screen_y, screen_offset_x)
    ax.plot_surface(screen_x, screen_y, screen_z, alpha=0.3, color='lightgray')
    
    # 각 파장별 광선 경로 시각화
    wavelengths = {
        'red': {'wavelength': 650, 'color': 'darkred', 'angles': [60, 72, 84]},
        'green': {'wavelength': 532, 'color': 'darkgreen', 'angles': [65, 75, 88]},
        'blue': {'wavelength': 445, 'color': 'blue', 'angles': [63, 73, 86]}
    }
    
    for wave_name, wave_info in wavelengths.items():
        for y_offset in [-0.1, 0, 0.1]:
            for angle in wave_info['angles']:
                # DMD에서 렌즈로
                ax.plot([dmd_position[0], lens_position[0]], 
                       [dmd_position[1]+y_offset, lens_position[1]+y_offset],
                       [dmd_position[2], lens_position[2]], 
                       '-', color=wave_info['color'], alpha=0.8, linewidth=1,
                       label=f'{wave_name.capitalize()} ({wave_info["wavelength"]}nm)' if y_offset == 0 else None)
                
                # 렌즈에서 미러로
                angle_rad = np.deg2rad(angle)
                mirror_x = mirror_position[0] + 2 * np.cos(angle_rad+np.pi)
                mirror_z = mirror_position[2] + 2 * np.sin(angle_rad+np.pi)
                
                ax.plot([lens_position[0], mirror_x], 
                       [lens_position[1]+y_offset, lens_position[1]+y_offset],
                       [lens_position[2], mirror_z], 
                       '-', color=wave_info['color'], alpha=0.8, linewidth=1)
                
                # 미러에서 스크린으로 (오프셋 적용)
                screen_z = screen_distance/2 + (angle - wave_info['angles'][0]) / (wave_info['angles'][-1] - wave_info['angles'][0]) * screen_distance * 0.4
                screen_y = screen_offset_y + (angle - wave_info['angles'][1]) * 0.1
                ax.plot([mirror_x, screen_offset_x], 
                       [lens_position[1]+y_offset, screen_y],
                       [mirror_z, screen_z], 
                       '-', color=wave_info['color'], alpha=0.8, linewidth=1)
    
    # 축 레이블 및 범례
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Ray Paths from DMD to Screen')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 시점 조정
    ax.view_init(elev=20, azim=-35)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"광선 경로 시각화가 {output_file}에 저장되었습니다.")
    
    return fig

# 예제 코드: 메인 코드에 아래 라인을 추가하면 렌더링이 실행됩니다
# 다양한 카메라 위치에서 렌더링
# camera_positions = [
#     (15, 15, 15, 5, 0, 15),   # 사선 위에서 바라보기
#     (0, 15, 5, 0, 0, 5),      # 측면에서 바라보기
#     (15, 0, 0, 5, 0, 15)      # 정면에서 바라보기
# ]
# render_scene_alternative(camera_positions, "projector_render")
# visualize_ray_paths_alternative(30, "projector_ray_paths.png")

def compare_te_tm_modes():
    """
    TE(s-편광)와 TM(p-편광) 모드의 반사율 및 투과율을 비교하는 그래프를 생성합니다.
    브루스터 각도에서의 특성도 표시합니다.
    """
    # 각도 범위 (0도에서 89도까지)
    angles_deg = np.linspace(0, 89, 90)
    angles_rad = np.radians(angles_deg)
    
    # 매질의 굴절률
    n1 = 1.0     # 공기
    n2 = 1.5     # 유리
    
    # TE(s-편광)과 TM(p-편광)의 반사 계수 계산
    r_TE = np.zeros_like(angles_rad, dtype=complex)
    r_TM = np.zeros_like(angles_rad, dtype=complex)
    
    for i, theta_i in enumerate(angles_rad):
        # 스넬의 법칙으로 투과각 계산
        theta_t = np.arcsin((n1 / n2) * np.sin(theta_i))
        
        # TE(s-편광) 반사 계수
        r_TE[i] = (n1*np.cos(theta_i) - n2*np.cos(theta_t)) / (n1*np.cos(theta_i) + n2*np.cos(theta_t))
        
        # TM(p-편광) 반사 계수
        r_TM[i] = (n2*np.cos(theta_i) - n1*np.cos(theta_t)) / (n2*np.cos(theta_i) + n1*np.cos(theta_t))
    
    # 반사율 계산 (복소 반사 계수의 크기 제곱)
    R_TE = np.abs(r_TE)**2
    R_TM = np.abs(r_TM)**2
    
    # 투과율 계산 (에너지 보존 법칙: T = 1 - R)
    T_TE = 1 - R_TE
    T_TM = 1 - R_TM
    
    # 브루스터 각도 계산 (tan(θ_B) = n2/n1)
    brewster_angle = np.degrees(np.arctan(n2/n1))
    print(f"브루스터 각도: {brewster_angle:.1f}°")
    
    # 브루스터 각도에서의 TM 모드 반사율
    brewster_idx = np.abs(angles_deg - brewster_angle).argmin()
    print(f"브루스터 각도({brewster_angle:.1f}°)에서의 TM 모드 반사율: {R_TM[brewster_idx]:.10f}")
    
    # 반사율 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, R_TE, 'r-', linewidth=2, label='TE 모드 (s-편광)')
    plt.plot(angles_deg, R_TM, 'b-', linewidth=2, label='TM 모드 (p-편광)')
    
    # 브루스터 각도 표시
    plt.axvline(x=brewster_angle, color='k', linestyle='--', alpha=0.6, label=f'브루스터 각도 ({brewster_angle:.1f}°)')
    
    plt.xlabel('입사각 (도)')
    plt.ylabel('반사율 (R)')
    plt.title('TE 및 TM 모드의 각도별 반사율 비교')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('reflection_comparison.png', dpi=300)
    
    # 투과율 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, T_TE, 'r-', linewidth=2, label='TE 모드 (s-편광)')
    plt.plot(angles_deg, T_TM, 'b-', linewidth=2, label='TM 모드 (p-편광)')
    
    # 브루스터 각도 표시
    plt.axvline(x=brewster_angle, color='k', linestyle='--', alpha=0.6, label=f'브루스터 각도 ({brewster_angle:.1f}°)')
    
    plt.xlabel('입사각 (도)')
    plt.ylabel('투과율 (T)')
    plt.title('TE 및 TM 모드의 각도별 투과율 비교')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('transmission_comparison.png', dpi=300)
    
    print("반사율 및 투과율 비교 그래프가 저장되었습니다.")

# 메인 실행 코드
if __name__ == "__main__":
    # 편광 분석 실행
    # analyze_polarization() 함수가 없으므로 주석 처리
    # analyze_polarization()
    
    # TE/TM 모드 비교 그래프 생성
    compare_te_tm_modes()
    
    # 프로젝터 기하학 시각화
    camera_positions = [
        (15, 15, 15, 5, 0, 15),   # 사선 위에서 바라보기
        (0, 15, 5, 0, 0, 5),      # 측면에서 바라보기
        (-5, 0, 15, 5, 0, 15)      # 정면에서 바라보기
    ]
    render_scene_alternative(camera_positions, "projector_render")
    visualize_ray_paths_alternative(30, "projector_ray_paths.png")
    
    # PolarizationAnalyzer 클래스 예제 실행
    run_polarization_analysis_example()
    
    print("모든 시뮬레이션이 완료되었습니다.")

def calculate_reflection_coefficients(layers, angle_deg, wavelength_nm):
    """
    전송 행렬 방법(TMM)을 사용하여 다층 코팅의 TE/TM 반사 계수 계산
    
    Parameters:
    -----------
    layers : list
        코팅 층 정보 (material, thickness)가 포함된 딕셔너리 리스트
    angle_deg : float
        입사각 (도)
    wavelength_nm : float
        파장 (나노미터)
        
    Returns:
    --------
    tuple (r_TE, r_TM) : TE 및 TM 모드의 반사 계수
    """
    # 공기의 굴절률
    n_air = 1.0
    
    # 입사각 (라디안 변환)
    theta_i = np.radians(angle_deg)
    
    # 재료별 굴절률 정의 (파장 의존성 단순화)
    material_index = {
        "Air": 1.0,
        "SiO2": 1.46,
        "TiO2": 2.5,
        "Ta2O5": 2.1,
        "Al2O3": 1.76,
        "Al": complex(0.9, 6.0),  # 알루미늄은 복소수 굴절률
        "Ag": complex(0.05, 4.2),  # 은
        "Au": complex(0.17, 3.1)   # 금
    }
    
    # TE 및 TM 모드에 대한 초기 전송 행렬
    M_TE = np.eye(2, dtype=complex)
    M_TM = np.eye(2, dtype=complex)
    
    # 각 층에 대한 전송 행렬 계산
    for layer in layers:
        material = layer["material"]
        thickness = layer["thickness"]
        
        # 재료 굴절률
        n = material_index.get(material, 1.5)  # 기본값 1.5
        
        # Snell's 법칙으로 각도 계산
        if isinstance(n, complex):
            # 복소수 굴절률인 경우 (금속)
            theta_t = np.arcsin(n_air * np.sin(theta_i) / abs(n))
        else:
            # 실수 굴절률인 경우 (유전체)
            theta_t = np.arcsin(n_air * np.sin(theta_i) / n)
        
        # 위상 계산
        beta = 2 * np.pi * n * thickness * np.cos(theta_t) / wavelength_nm
        
        # TE 모드 임피던스
        eta_TE = n * np.cos(theta_t)
        
        # TM 모드 임피던스
        eta_TM = n / np.cos(theta_t)
        
        # TE 모드 전송 행렬
        M11_TE = np.cos(beta)
        M12_TE = 1j * np.sin(beta) / eta_TE
        M21_TE = 1j * np.sin(beta) * eta_TE
        M22_TE = np.cos(beta)
        
        M_layer_TE = np.array([[M11_TE, M12_TE], [M21_TE, M22_TE]])
        M_TE = np.dot(M_TE, M_layer_TE)
        
        # TM 모드 전송 행렬
        M11_TM = np.cos(beta)
        M12_TM = 1j * np.sin(beta) / eta_TM
        M21_TM = 1j * np.sin(beta) * eta_TM
        M22_TM = np.cos(beta)
        
        M_layer_TM = np.array([[M11_TM, M12_TM], [M21_TM, M22_TM]])
        M_TM = np.dot(M_TM, M_layer_TM)
    
    # TE 모드 반사 계수
    eta0_TE = n_air * np.cos(theta_i)
    etaN_TE = material_index.get("Al", complex(0.9, 6.0)) # 마지막 층 (기판)
    
    Y_TE = (M_TE[0, 0] + M_TE[0, 1] * etaN_TE) / (M_TE[1, 0] + M_TE[1, 1] * etaN_TE)
    r_TE = (eta0_TE - Y_TE) / (eta0_TE + Y_TE)
    
    # TM 모드 반사 계수
    eta0_TM = n_air / np.cos(theta_i)
    etaN_TM = material_index.get("Al", complex(0.9, 6.0))
    
    Y_TM = (M_TM[0, 0] + M_TM[0, 1] * etaN_TM) / (M_TM[1, 0] + M_TM[1, 1] * etaN_TM)
    r_TM = (eta0_TM - Y_TM) / (eta0_TM + Y_TM)
    
    return r_TE, r_TM

def simulate_advanced_coating(angle_deg, wavelength_nm, coating_type="multilayer"):
    # 코팅 유형에 따른 파라미터 설정
    if coating_type == "multilayer":
        layers = [
            {"material": "TiO2", "thickness": wavelength_nm/4/2.5},  # n≈2.5
            {"material": "SiO2", "thickness": wavelength_nm/4/1.46}, # n≈1.46
            # 여러 층 반복...
        ]
    elif coating_type == "hybrid":
        layers = [
            {"material": "Al", "thickness": 100},  # 기본 금속층
            {"material": "SiO2", "thickness": wavelength_nm/4/1.46},
            {"material": "TiO2", "thickness": wavelength_nm/4/2.5},
            # 추가 층...
        ]
    
    # 입사각에 따른 TE/TM 반사율 계산
    r_TE, r_TM = calculate_reflection_coefficients(layers, angle_deg, wavelength_nm)
    
    return {
        "R_TE": abs(r_TE)**2,
        "R_TM": abs(r_TM)**2,
        "R_avg": (abs(r_TE)**2 + abs(r_TM)**2) / 2,
        "R_diff": abs(abs(r_TE)**2 - abs(r_TM)**2)
    }

def visualize_coating_performance(mirror_profile, coating_type="angle_invariant"):
    angles = np.arange(20, 80, 5)
    wavelengths = [450, 550, 650]  # RGB 대표 파장
    
    results = {}
    for wl in wavelengths:
        results[wl] = {}
        for angle in angles:
            results[wl][angle] = simulate_advanced_coating(angle, wl, coating_type)
    
    # 시각화 코드...
    return results

# ... existing code ...

# 메인 코드 부분에 추가 (if __name__ == "__main__": 아래에 추가)
if __name__ == "__main__":
    # ... existing code ...
    
    # 코팅 시뮬레이션 부분 추가
    print("\n코팅 시뮬레이션 실행 중...\n")
    
    # 간단한 비구면 미러 프로파일 정의 (예시)
    mirror_profile = {
        "center": {"position": 0.0, "angle": 45.0},
        "edge": {"position": 1.0, "angle": 70.0},
        "shape": "parabolic"
    }
    
    # 다양한 코팅 유형에 대한 성능 비교
    print("일반 다층 코팅 시뮬레이션:")
    multilayer_results = visualize_coating_performance(mirror_profile, "multilayer")
    
    print("\n하이브리드 코팅 시뮬레이션:")
    hybrid_results = visualize_coating_performance(mirror_profile, "hybrid")
    
    # 그래프로 시각화
    plt.figure(figsize=(14, 8))
    
    # 각도에 따른 TE/TM 반사율 차이 비교 (550nm 파장 기준)
    angles = np.arange(20, 80, 5)
    
    # 다층 코팅 결과
    te_refl_ml = [multilayer_results[550][angle]["R_TE"] for angle in angles]
    tm_refl_ml = [multilayer_results[550][angle]["R_TM"] for angle in angles]
    diff_ml = [multilayer_results[550][angle]["R_diff"] for angle in angles]
    
    # 하이브리드 코팅 결과
    te_refl_hb = [hybrid_results[550][angle]["R_TE"] for angle in angles]
    tm_refl_hb = [hybrid_results[550][angle]["R_TM"] for angle in angles]
    diff_hb = [hybrid_results[550][angle]["R_diff"] for angle in angles]
    
    # 플롯 생성
    plt.subplot(2, 1, 1)
    plt.plot(angles, te_refl_ml, 'b-', label='TE (Multilayer)')
    plt.plot(angles, tm_refl_ml, 'r-', label='TM (Multilayer)')
    plt.plot(angles, te_refl_hb, 'b--', label='TE (Hybrid)')
    plt.plot(angles, tm_refl_hb, 'r--', label='TM (Hybrid)')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Reflectance')
    plt.title('TE/TM Mode Reflectance Comparison (550nm)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(angles, diff_ml, 'g-', label='TE-TM Difference (Multilayer)')
    plt.plot(angles, diff_hb, 'g--', label='TE-TM Difference (Hybrid)')
    plt.axhline(y=0.05, color='r', linestyle='-', alpha=0.3, label='Threshold 5%')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('TE-TM Reflectance Difference')
    plt.title('Polarization Dependence Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("모든 시뮬레이션이 완료되었습니다.")

# 광선 경로 - 3D 공간에서 다른 색상으로
wavelengths = {
    'red': {'wavelength': 650, 'color': 'darkred', 'angles': [60, 72, 84]},
    'green': {'wavelength': 532, 'color': 'darkgreen', 'angles': [65, 75, 88]},
    'blue': {'wavelength': 445, 'color': 'blue', 'angles': [63, 73, 86]}
}

# 각 파장별 광선 경로 시각화
for wave_name, wave_info in wavelengths.items():
    for y_offset in [-0.1, 0, 0.1]:  # Y축 오프셋으로 광선 구분
        for angle in wave_info['angles']:
            # DMD에서 렌즈로
            ax.plot([dmd_pos[0], lens_pos[0]], 
                   [dmd_pos[1]+y_offset, lens_pos[1]+y_offset],
                   [dmd_pos[2], lens_pos[2]], 
                   '-', color=wave_info['color'], alpha=0.8, linewidth=1,
                   label=f'{wave_name.capitalize()} ({wave_info["wavelength"]}nm)' if y_offset == 0 else None)
            
            # 렌즈에서 미러로
            angle_rad = np.deg2rad(angle)
            mirror_x = mirror_center[0] + mirror_radius * np.cos(angle_rad+np.pi)
            mirror_z = mirror_center[2] + mirror_radius * np.sin(angle_rad+np.pi)
            
            ax.plot([lens_pos[0], mirror_x], 
                   [lens_pos[1]+y_offset, lens_pos[1]+y_offset],
                   [lens_pos[2], mirror_z], 
                   '-', color=wave_info['color'], alpha=0.8, linewidth=1)
            
            # 미러에서 Y-Z 평면 스크린으로 반사
            screen_z = screen_size/2 + (angle - wave_info['angles'][0]) / (wave_info['angles'][-1] - wave_info['angles'][0]) * screen_size * 0.4
            ax.plot([mirror_x, yz_screen_x], 
                   [lens_pos[1]+y_offset, lens_pos[1]+y_offset + (angle - wave_info['angles'][1]) * 0.01],
                   [mirror_z, screen_z], 
                   '-', color=wave_info['color'], alpha=0.8, linewidth=1)

# 범례 추가
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

def simulate_color_mixing_yz(screen_size=1.0, resolution=100):
    """
    Y-Z 평면에서의 RGB 합성 색상차 시뮬레이션 (균일한 조명 조건)
    :param screen_size: 스크린 크기
    :param resolution: 시뮬레이션 해상도
    """
    # Y-Z 평면 그리드 생성
    y = np.linspace(-screen_size/2, screen_size/2, resolution)
    z = np.linspace(0, screen_size, resolution)
    Y, Z = np.meshgrid(y, z)
    
    # 각 파장별 강도 분포 초기화
    intensity_r = np.zeros((resolution, resolution))
    intensity_g = np.zeros((resolution, resolution))
    intensity_b = np.zeros((resolution, resolution))
    
    # 연속적인 각도 분포 정의
    angle_range = np.linspace(60, 85, 100)  # 더 조밀한 각도 분포
    
    # 각 파장별 가우시안 빔 프로파일 생성
    for wave_name in ['red', 'green', 'blue']:
        # 가우시안 빔 파라미터
        beam_width = screen_size/4  # 빔 너비
        intensity_peak = 1.0  # 최대 강도
        
        for angle in angle_range:
            # 중심점 계산 (연속적인 분포)
            center_z = screen_size/2 + (angle - 60) / (85 - 60) * screen_size * 0.4
            center_y = 0  # 중심축 정렬
            
            # 가우시안 빔 프로파일
            gaussian = intensity_peak * np.exp(
                -2 * ((Y - center_y)**2 + (Z - center_z)**2) / beam_width**2
            )
            
            # 각도에 따른 가중치 (더 자연스러운 분포를 위해)
            angle_weight = np.exp(-(angle - 72.5)**2 / (10**2))
            
            # 해당 파장의 강도 분포에 추가
            if wave_name == 'red':
                intensity_r += gaussian * angle_weight / len(angle_range)
            elif wave_name == 'green':
                intensity_g += gaussian * angle_weight / len(angle_range)
            else:  # blue
                intensity_b += gaussian * angle_weight / len(angle_range)
    
    # 전체 강도 정규화 및 균일화
    def normalize_and_uniform(intensity, uniformity_factor=0.8):
        # 강도 정규화
        intensity = intensity / np.max(intensity)
        # 균일화 적용
        intensity = intensity * uniformity_factor + (1 - uniformity_factor)
        return np.clip(intensity, 0, 1)
    
    intensity_r = normalize_and_uniform(intensity_r)
    intensity_g = normalize_and_uniform(intensity_g)
    intensity_b = normalize_and_uniform(intensity_b)
    
    # RGB 이미지 생성
    rgb_image = np.stack([intensity_r, intensity_g, intensity_b], axis=2)
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # RGB 합성 이미지
    plt.subplot(131)
    plt.imshow(rgb_image, extent=[-screen_size/2, screen_size/2, 0, screen_size])
    plt.colorbar(label='Normalized Intensity')
    plt.xlabel('Y position (m)')
    plt.ylabel('Z position (m)')
    plt.title('RGB Color Mixing (Uniform Source)')
    
    # 수평 프로파일 (Z = screen_size/2에서)
    plt.subplot(132)
    z_idx = resolution // 2
    plt.plot(y, intensity_r[z_idx], 'r-', label='Red', alpha=0.7)
    plt.plot(y, intensity_g[z_idx], 'g-', label='Green', alpha=0.7)
    plt.plot(y, intensity_b[z_idx], 'b-', label='Blue', alpha=0.7)
    plt.xlabel('Y position (m)')
    plt.ylabel('Normalized Intensity')
    plt.title('Horizontal Intensity Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 수직 프로파일 (Y = 0에서)
    plt.subplot(133)
    y_idx = resolution // 2
    plt.plot(z, intensity_r[:, y_idx], 'r-', label='Red', alpha=0.7)
    plt.plot(z, intensity_g[:, y_idx], 'g-', label='Green', alpha=0.7)
    plt.plot(z, intensity_b[:, y_idx], 'b-', label='Blue', alpha=0.7)
    plt.xlabel('Z position (m)')
    plt.ylabel('Normalized Intensity')
    plt.title('Vertical Intensity Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('uniform_color_mixing_yz.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 색상차 분석
    plt.figure(figsize=(15, 5))
    
    # Red-Green 색상차
    plt.subplot(131)
    color_diff_rg = np.abs(intensity_r - intensity_g)
    plt.imshow(color_diff_rg, extent=[-screen_size/2, screen_size/2, 0, screen_size], cmap='RdYlBu')
    plt.colorbar(label='R-G Difference')
    plt.xlabel('Y position (m)')
    plt.ylabel('Z position (m)')
    plt.title('Red-Green Color Difference')
    
    # Red-Blue 색상차
    plt.subplot(132)
    color_diff_rb = np.abs(intensity_r - intensity_b)
    plt.imshow(color_diff_rb, extent=[-screen_size/2, screen_size/2, 0, screen_size], cmap='RdYlBu')
    plt.colorbar(label='R-B Difference')
    plt.xlabel('Y position (m)')
    plt.ylabel('Z position (m)')
    plt.title('Red-Blue Color Difference')
    
    # Green-Blue 색상차
    plt.subplot(133)
    color_diff_gb = np.abs(intensity_g - intensity_b)
    plt.imshow(color_diff_gb, extent=[-screen_size/2, screen_size/2, 0, screen_size], cmap='RdYlBu')
    plt.colorbar(label='G-B Difference')
    plt.xlabel('Y position (m)')
    plt.ylabel('Z position (m)')
    plt.title('Green-Blue Color Difference')
    
    plt.tight_layout()
    plt.savefig('uniform_color_difference_yz.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("균일한 광원을 사용한 Y-Z 평면 색상 시뮬레이션이 완료되었습니다.")

def main():
    """
    메인 실행 함수
    """
    # 편광 분석 예제 실행
    run_polarization_analysis_example()
    
    # 3D 시각화
    camera_positions = [
        (15, 15, 15, 5, 0, 15),   # 사선 위에서 바라보기
        (0, 15, 5, 0, 0, 5),      # 측면에서 바라보기
        (15, 0, 0, 5, 0, 15)      # 정면에서 바라보기
    ]
    render_scene_alternative(camera_positions, "projector_render")
    visualize_ray_paths_alternative(30, "projector_ray_paths.png")
    
    # Y-Z 평면 색상 시뮬레이션
    print("\nY-Z 평면 색상 시뮬레이션 시작...")
    simulate_color_mixing_yz(screen_size=1.0, resolution=200)
    print("Y-Z 평면 색상 시뮬레이션 완료\n")
    
    # TE/TM 모드 비교
    compare_te_tm_modes()
    
    # 코팅 시뮬레이션
    print("\n코팅 시뮬레이션 실행 중...")
    simulate_advanced_coating(45, 550, "multilayer")
    simulate_advanced_coating(45, 550, "hybrid")
    print("모든 시뮬레이션이 완료되었습니다.")

if __name__ == "__main__":
    main()