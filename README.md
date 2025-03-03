# polarization

이 문서는 `polarization.py` 파일의 사용법과 코드 설명을 제공합니다. 이 코드는 초단초점 프로젝터의 광학 시뮬레이션을 수행하며, 편광 효과와 색상 편차를 분석하는 데 사용됩니다.

## 사용법

1. **환경 설정**: 
   - 이 코드는 `raysect` 라이브러리를 사용하므로, 해당 라이브러리를 설치해야 합니다. 
   - Python 3.x 환경에서 실행해야 하며, 필요한 패키지를 설치합니다.
   ```bash
   pip install raysect matplotlib numpy
   ```

2. **코드 실행**:
   - `polarization.py` 파일을 실행하면, 광학 시뮬레이션이 시작됩니다. 
   - 메인 함수인 `main()`이 호출되어 다양한 시뮬레이션을 수행합니다.

3. **출력 결과**:
   - 시뮬레이션 결과는 여러 개의 PNG 파일로 저장됩니다. 
   - 예를 들어, `projector_render_camera1.png`, `color_simulation_selective.png` 등의 파일이 생성됩니다.

## 코드 설명

### 주요 클래스 및 함수

- **`PolarizationAnalyzer` 클래스**:
  - 편광 상태 분석 및 시뮬레이션을 위한 클래스입니다.
  - 초기 편광 상태를 정의하고, 주어진 각도에서의 반사 강도를 계산합니다.

- **`simulate_color_mixing_yz` 함수**:
  - Y-Z 평면에서 RGB 색상 혼합을 시뮬레이션합니다.
  - 가우시안 빔 프로파일을 사용하여 각 파장별 강도를 계산합니다.

- **`calculate_reflection_coefficients` 함수**:
  - 전송 행렬 방법(TMM)을 사용하여 다층 코팅의 TE/TM 반사 계수를 계산합니다.

### 수정해야 할 코드

1. **광원 설정**:
   - `wavelengths` 딕셔너리에서 각 색상의 파장을 수정할 수 있습니다.
   ```python
   wavelengths = {
       'red': {'wavelength': 650, 'color': 'darkred', 'angles': [60, 72, 84]},
       'green': {'wavelength': 532, 'color': 'darkgreen', 'angles': [65, 75, 88]},
       'blue': {'wavelength': 445, 'color': 'blue', 'angles': [63, 73, 86]}
   }
   ```

2. **스크린 오프셋 조정**:
   - 스크린의 위치를 조정하려면 `screen_offset_x` 및 `screen_offset_y` 변수를 수정합니다.
   ```python
   screen_offset_x = -5  # 스크린 X축 오프셋
   screen_offset_y = 2   # 스크린 Y축 오프셋
   ```

3. **시뮬레이션 해상도 변경**:
   - `simulate_color_mixing_yz` 함수의 `resolution` 매개변수를 변경하여 시뮬레이션 해상도를 조정할 수 있습니다.
   ```python
   simulate_color_mixing_yz(screen_size=1.0, resolution=200)
   ```

## 결과

- 이 코드는 초단초점 프로젝터의 색상 편차 및 밝기 불균일성을 시뮬레이션하고, 편광 효과가 색상 및 밝기 분포에 미치는 영향을 분석합니다.
- 결과는 다양한 그래프와 이미지를 통해 시각화됩니다.

## 참고

- 추가적인 기능이나 수정이 필요할 경우, 코드 내 주석을 참고하여 필요한 부분을 수정할 수 있습니다.
- 각 함수와 클래스에 대한 자세한 설명은 코드 내 주석을 통해 확인할 수 있습니다.
