# polarization

이 문서는 `polarization.py` 파일의 사용법과 코드 설명을 제공합니다. 이 코드는 초단초점 프로젝터의 광학 시뮬레이션을 수행하며, 편광 효과와 색상 편차를 분석하는 데 사용됩니다.

## 목차
- [설치 및 환경 설정](#설치-및-환경-설정)
- [기본 사용법](#기본-사용법)
- [주요 클래스 및 함수](#주요-클래스-및-함수)
- [시뮬레이션 함수](#시뮬레이션-함수)
- [시각화 함수](#시각화-함수)
- [코드 수정 방법](#코드-수정-방법)
- [결과 해석](#결과-해석)
- [고급 사용법](#고급-사용법)
- [참고사항](#참고사항)

## 설치 및 환경 설정
이 코드를 실행하기 위해 필요한 패키지와 설정은 다음과 같습니다:
- Python 3.7 이상
- 필수 패키지: raysect, matplotlib, numpy, scipy
```bash
pip install raysect matplotlib numpy scipy
```

## 기본 사용법
### 코드 실행
```bash
python polarization.py
```
이 명령어로 코드를 실행하면 `main()` 함수가 호출되어 다양한 시뮬레이션과 시각화가 수행됩니다.

### 특정 시뮬레이션만 실행
특정 시뮬레이션만 실행하려면 코드 내에서 `main()` 함수를 수정하거나, 새로운 스크립트에서 다음과 같이 사용할 수 있습니다:
```python
import polarization

# 특정 기능만 실행
polarization.visualize_ray_paths(num_rays=30, output_file="custom_rays.png")
polarization.simulate_color_mixing_yz(screen_size=1.5, resolution=150)
```

## 주요 클래스 및 함수

### `Point3D` 클래스
3차원 공간의 점을 표현하는 클래스로, 광선 경로 계산에 사용됩니다.
```python
point = Point3D(x=1.0, y=2.0, z=3.0)
x_value = point[0]  # 인덱스로 접근 가능
```

### `PolarizationAnalyzer` 클래스
편광 상태 분석 및 시뮬레이션을 위한 클래스로, 다음 기능을 제공합니다:
- 초기 편광 상태 정의
- 다양한 각도에서의 반사 강도 계산
- 스토크스 매개변수 계산
- 결과 시각화

```python
analyzer = PolarizationAnalyzer()
results = analyzer.analyze_angle_range(
    angles_deg=range(0, 90, 5), 
    apply_hwp=True
)
analyzer.plot_analysis_results(results, title="편광 분석 결과")
```

### `fresnel_coeffs` 함수
입사각과 파장을 기반으로 프레넬 계수를 계산합니다.
```python
r_s, r_p = fresnel_coeffs(angle_deg=45, wavelength_nm=550)
```

### `calc_stokes` 함수
존스 벡터로부터 스토크스 매개변수를 계산합니다.
```python
stokes = calc_stokes(jones_vector)
```

## 시뮬레이션 함수

### `propagate_with_hwp` / `propagate_without_hwp` 함수
편광 상태가 HWP(Half-Wave Plate)를 통과하거나 통과하지 않고 전파될 때의 효과를 시뮬레이션합니다.
```python
# 초기 존스 벡터 (수평 편광)
jones_in = np.array([1.0, 0.0], dtype=complex)
# HWP 없이 전파
jones_out_no_hwp = propagate_without_hwp(jones_in, incidence_angle=45)
# HWP와 함께 전파
jones_out_hwp = propagate_with_hwp(jones_in, incidence_angle=45)
```

### `propagate_with_selective_hwp` 함수
특정 색상(파장)에 대해 선택적으로 HWP를 적용하여 전파 효과를 시뮬레이션합니다.
```python
jones_out_red = propagate_with_selective_hwp(
    jones_in, 
    incidence_angle=45, 
    color_name='red'
)
```

### `simulate_color_mixing_yz` 함수
Y-Z 평면에서 RGB 색상 혼합을 시뮬레이션하고, 그 결과를 시각화합니다.
```python
simulate_color_mixing_yz(screen_size=1.0, resolution=100)
```

### `simulate_advanced_coating` 함수
다양한 코팅 유형(다층, 메타 표면 등)에 대한 반사 특성을 시뮬레이션합니다.
```python
coating_results = simulate_advanced_coating(
    angle_deg=45, 
    wavelength_nm=550, 
    coating_type="multilayer"
)
```

### `calculate_reflection_coefficients` 함수
전송 행렬 방법(TMM)을 사용하여 다층 구조의 반사 계수를 계산합니다.
```python
layers = [
    {'n': 1.0, 'd': 0},          # 공기
    {'n': 1.5, 'd': 120},        # 120nm 두께의 SiO2
    {'n': 0.055+3.32j, 'd': 0}   # 알루미늄 기판
]
r_s, r_p = calculate_reflection_coefficients(
    layers, 
    angle_deg=45, 
    wavelength_nm=550
)
```

## 시각화 함수

### `visualize_ray_paths` 함수
DMD에서 스크린까지의 광선 경로를 시각화합니다.
```python
visualize_ray_paths(num_rays=20, output_file="ray_paths.png")
```

### `render_scene` 함수
다양한 카메라 위치에서 전체 광학 시스템을 3D로 렌더링합니다.
```python
camera_positions = [
    {'pos': [5, 0, 0], 'target': [0, 0, 0]},
    {'pos': [0, 5, 0], 'target': [0, 0, 0]}
]
render_scene(camera_positions, output_prefix="custom_render")
```

### `plot_polarization_state` 함수
존스 벡터로 표현된 편광 상태를 시각화합니다.
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_polarization_state(jones_vector, ax, title="편광 상태")
plt.show()
```

### `plot_poincare_sphere` 함수
포앵카레 구(Poincaré sphere)에 여러 편광 상태를 시각화합니다.
```python
stokes_list = [calc_stokes(jones1), calc_stokes(jones2)]
colors = ['red', 'blue']
labels = ['초기 상태', '반사 후 상태']
plot_poincare_sphere(stokes_list, colors, labels, title="편광 상태 변화")
```

### `plot_stokes_comparison` 함수
반사 전후의 스토크스 매개변수 변화를 비교 시각화합니다.
```python
plot_stokes_comparison(
    stokes_before, 
    stokes_after, 
    labels=['반사 전', '반사 후'], 
    title="반사에 의한 편광 변화"
)
```

### `visualize_coating_performance` 함수
다양한 코팅 유형에 대한 성능을 시각화합니다.
```python
mirror_profile = {
    'angles': range(0, 90, 1),
    'reflectances': {
        'TE': [...],  # TE 모드 반사율
        'TM': [...]   # TM 모드 반사율
    }
}
visualize_coating_performance(
    mirror_profile, 
    coating_type="angle_invariant"
)
```

## 코드 수정 방법

### 1. 광원 설정 수정
각 색상의 파장과 반사 각도를 수정할 수 있습니다.
```python
wavelengths = {
    'red': {'wavelength': 650, 'color': 'darkred', 'angles': [60, 72, 84]},
    'green': {'wavelength': 532, 'color': 'darkgreen', 'angles': [65, 75, 88]},
    'blue': {'wavelength': 445, 'color': 'blue', 'angles': [63, 73, 86]}
}
```

### 2. 스크린 오프셋 조정
스크린의 위치를 조정하려면 오프셋 변수를 수정합니다.
```python
screen_offset_x = -5  # 스크린 X축 오프셋
screen_offset_y = 2   # 스크린 Y축 오프셋
```

### 3. 시뮬레이션 해상도 변경
시뮬레이션 해상도를 조정하여 정밀도와 계산 시간을 조절할 수 있습니다.
```python
simulate_color_mixing_yz(screen_size=1.0, resolution=200)
```

### 4. 광 강도 정규화 및 균일도 조정
광 강도의 정규화와 균일도 조정을 위한 함수 매개변수를 수정할 수 있습니다.
```python
def normalize_and_uniform(intensity, uniformity_factor=0.8):
    # 강도 정규화 및 균일도 조정
    # uniformity_factor가 1에 가까울수록 더 균일한 분포
```

## 결과 해석

### 색상 혼합 결과
- `color_mixing_yz.png`: Y-Z 평면에서의 RGB 색상 혼합 결과
- `color_simulation_selective.png`: 선택적 HWP 적용 시 색상 변화
- `color_difference_yz.png`: 색상 채널 간 차이 시각화

### 편광 분석 결과
- 포앵카레 구 시각화: 3D 공간에서의 편광 상태 변화
- 편광 타원 플롯: 2D 평면에서의 전기장 진동 방향

### 반사 계수 분석
- TE/TM 모드 반사 계수: 입사각에 따른 변화 그래프
- 코팅 성능 그래프: 다양한 코팅 유형에 따른 반사 특성

## 고급 사용법

### 맞춤형 미러 코팅 시뮬레이션
```python
def custom_coating_simulation():
    # 사용자 정의 코팅 설계
    custom_layers = [
        {'n': 1.0, 'd': 0},          # 공기
        {'n': 1.38, 'd': 110},       # MgF2 코팅 (110nm)
        {'n': 2.3, 'd': 55},         # TiO2 코팅 (55nm)
        {'n': 0.055+3.32j, 'd': 0}   # 알루미늄 기판
    ]
    
    # 다양한 입사각에서 반사 계수 계산
    angles = range(0, 90, 1)
    r_s_values = []
    r_p_values = []
    
    for angle in angles:
        r_s, r_p = calculate_reflection_coefficients(
            custom_layers, 
            angle_deg=angle, 
            wavelength_nm=550
        )
        r_s_values.append(abs(r_s)**2)
        r_p_values.append(abs(r_p)**2)
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(angles, r_s_values, 'b-', label='TE mode')
    plt.plot(angles, r_p_values, 'r-', label='TM mode')
    plt.xlabel('입사각 (도)')
    plt.ylabel('반사율')
    plt.title('맞춤형 미러 코팅의 반사 특성')
    plt.legend()
    plt.grid(True)
    plt.savefig('custom_coating_reflectance.png', dpi=300)
    plt.close()
```
사용자 정의 코팅 설계를 통해 다양한 입사각에서의 반사 특성을 분석할 수 있습니다.

### 다양한 입사광 분포 시뮬레이션
```python
def simulate_with_custom_distribution():
    # 사용자 정의 입사광 분포
    def custom_beam_profile(y, z, beam_center_y, beam_center_z, beam_width):
        # Top-hat 프로파일 (균일한 원형 빔)
        r = np.sqrt((y - beam_center_y)**2 + (z - beam_center_z)**2)
        return np.where(r <= beam_width/2, 1.0, 0.0)
    
    # 시뮬레이션 수행
    # (코드 구현 필요)
```
Top-hat 프로파일과 같은 사용자 정의 빔 프로파일을 적용하여 시뮬레이션할 수 있습니다.

## 참고사항

### 실행 시간 단축
- 대규모 시뮬레이션의 경우 `resolution` 매개변수를 낮추어 실행 시간을 단축할 수 있습니다.
- 메인 함수에서 필요 없는 시뮬레이션을 주석 처리하여 특정 분석만 수행할 수 있습니다.

### 문제 해결
- `ImportError`: 필요한 패키지가 설치되어 있는지 확인하세요.
- 메모리 오류: 해상도를 낮추거나, 광선 수를 줄여보세요.
- 렌더링 오류: 렌더링 함수의 카메라 위치 매개변수를 조정해보세요.

### 참고 문헌
- M. Born and E. Wolf, "Principles of Optics"
- E. Hecht, "Optics"
- D. Goldstein, "Polarized Light"
```