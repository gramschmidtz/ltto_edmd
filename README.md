# DMD, EDMD, Galerkin 방법을 이용한 저추력 궤도 최적화

이 프로젝트는 우주선의 저추력 궤도 최적화를 위해 DMD, EDMD, Galerkin 방법을 적용하는 데 중점을 둡니다. 우주선 동역학을 모델링하고, 데이터셋을 생성하며, 시스템 식별 및 궤도 예측을 위해 DMD/EDMD를 적용하는 파이썬 기반 프레임워크를 제공합니다.

## 프로젝트 개요

이 프로젝트의 핵심은 데이터 기반 방법, 특히 EDMD를 사용하여 저추력 우주선의 동역학을 근사하는 것입니다. 이를 통해 더 높은 차원의 관측 가능 공간에서 비선형 우주선 동역학의 선형 모델을 생성할 수 있으며, 이는 궤도 최적화 및 제어에 사용될 수 있습니다.

이 프로젝트에는 다음을 위한 모듈이 포함되어 있습니다:
- 우주선의 운동 방정식 정의.
- EDMD 모델 훈련을 위한 궤도 데이터 생성.
- 시스템 행렬을 식별하기 위한 DMD 구현.
- 학습된 모델을 기반으로 궤도를 시뮬레이션하고 시각화.

## 설치

이 프로젝트는 의존성 관리를 위해 [Poetry](https://python-poetry.org/)를 사용합니다.

1.  **저장소 복제:**
    ```bash
    git clone https://github.com/gramschmidtz/ltto_edmd.git
    cd ltto_edmd
    ```

2.  **Poetry를 사용하여 의존성 설치:**
    ```bash
    poetry install
    ```
    이렇게 하면 가상 환경이 생성되고 `pyproject.toml`에 나열된 `numpy`, `scipy`, `matplotlib`, GPU 가속을 위한 `cupy` 등 모든 필수 패키지가 설치됩니다.

## 실행 방법

`scripts` 디렉토리에는 시뮬레이션을 실행하기 위한 메인 스크립트가 포함되어 있습니다.

### DMD 시뮬레이션 실행

데이터셋 생성, DMD 피팅 및 궤도 롤아웃을 포함하는 DMD 시뮬레이션을 실행하려면 다음 명령을 실행하십시오:

```bash
poetry run python scripts/DMD.py
```

이 스크립트는 다음을 수행합니다:
1.  궤도 데이터셋을 구축합니다.
2.  DMD를 사용하여 선형 모델(A 및 B 행렬)을 피팅합니다.
3.  학습된 모델과 미리 정의된 제어 프로필을 사용하여 새로운 궤도를 롤아웃합니다.
4.  궤도 플롯을 생성하고 저장합니다 (`fig/DMD_result.png`).

`src/dynamics/config.py`에서 궤도 수 및 시뮬레이션 시간과 같은 시뮬레이션 매개변수를 수정할 수 있습니다.

## 프로젝트 구조

```
.
├── .gitignore
├── poetry.lock
├── pyproject.toml
├── README.md
├── fig/
├── scripts/
│   ├── DMD.py              # DMD 시뮬레이션을 위한 메인 스크립트
│   └── ground_truth.py     # (진행 중) 실제 값 비교용
└── src/
    ├── controllers/
    │   └── test_controller.py # 테스트 제어 프로필 정의
    ├── dynamics/
    │   ├── config.py          # 동역학 및 시뮬레이션 구성
    │   ├── discrete_dynamics.py # RK4를 사용한 이산 동역학
    │   └── dynamics_reduced.py  # 축소된 차수의 동역학 모델
    └── edmd/
        ├── make_dataset.py    # 궤도 데이터셋 생성을 위한 함수
        └── observables.py     # (진행 중) EDMD 관측 가능 항목 정의용
```

## 핵심 구성 요소

-   **`src/dynamics`**: 우주선의 운동 방정식 구현을 포함합니다.
-   **`src/edmd`**: 데이터셋 생성을 위한 도구를 포함하며, 관측 가능 함수의 선택과 같은 EDMD 관련 로직을 담기 위한 것입니다.
-   **`src/controllers`**: 우주선의 제어 입력을 정의하는 데 사용됩니다.
-   **`scripts`**: 시뮬레이션 및 실험 실행을 위한 상위 수준 스크립트.
-   **`pyproject.toml`**: Poetry를 위한 프로젝트 의존성 및 메타데이터를 정의합니다.
-   **`fig`**: 생성된 플롯을 저장하기 위한 기본 디렉토리.