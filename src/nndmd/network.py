# src/nndmd/network.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU, x_scale=1.0):
        super().__init__()

        latent_dim = output_dim - input_dim

        self.register_buffer('x_scale', torch.tensor(x_scale))

        layers = [
            nn.Linear(input_dim, 32),
            activation(),
            nn.Linear(32, 64),
            activation(),
            nn.Linear(64, latent_dim)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_scaled = x / self.x_scale
        latent = self.net(x_scaled)
        combined = torch.cat((x_scaled, latent), dim=1)   # 최종 output_dim 유지
        return combined

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU, x_scale=1.0):
        super().__init__()

        self.register_buffer('x_scale', torch.tensor(x_scale))

        layers = [
            nn.Linear(input_dim, 128),
            activation(),
            nn.Linear(128, 64),
            activation(),
            nn.Linear(64, output_dim)
        ]

        self.net = nn.Sequential(*layers)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        z = self.net(x)
        out = self.final_activation(z)
        return out * self.x_scale

class NNDMD_Model(nn.Module):
    """
    Encoder, Decoder, A, B를 모두 포함하는 통합 모델
    입력 정규화(U scaling)까지 내부에서 처리함
    """
    def __init__(self, state_dim, input_dim, latent_dim, u_scale=1.0):
        super().__init__()
        
        # 하위 모듈 생성
        self.encoder = Encoder(state_dim, latent_dim)
        self.decoder = Decoder(latent_dim, state_dim)
        
        # 행렬 A, B 등록
        self.A = nn.Parameter(torch.eye(latent_dim))
        self.B = nn.Parameter(torch.zeros(latent_dim, input_dim))
        
        # [핵심] U 스케일링 상수 저장 (buffer)
        # 예: 1e7을 저장해두면, 들어오는 u에 자동으로 곱해줌
        self.register_buffer('u_scale', torch.tensor(u_scale))

    def get_latent(self, x):
        return self.encoder(x)

    def predict_next(self, z, u):
        """
        z_next = A*z + B*(u * scale)
        """
        # 여기서 u를 자동으로 뻥튀기해줌!
        u_scaled = u * self.u_scale 
        
        # 선형 동역학 계산
        z_next = torch.einsum('ij,bj->bi', self.A, z) + \
                 torch.einsum('ij,bj->bi', self.B, u_scaled)
        return z_next
    
    def decode(self, z):
        return self.decoder(z)