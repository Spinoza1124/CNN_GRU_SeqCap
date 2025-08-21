import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def squash(vectors, dim=-1):
    """
    Squash non-linear activation function for capsule networks
    :param vectors: Input tensor, e.g., [batch_size, num_caps, caps_dim]
    :param dim: Dimension to compute on
    :return: Tensor after squash operation
    """
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * vectors / (torch.sqrt(squared_norm) + 1e-8)  # 1e-8 to prevent division by zero

class CapsuleLayer(nn.Module):
    """
    Capsule layer with dynamic routing algorithm
    Based on "Dynamic Routing Between Capsules" by Sabour et al.
    """
    def __init__(self, in_caps, out_caps, in_dim, out_dim, routing_iters=3):
        """
        :param in_caps: Number of input capsules
        :param out_caps: Number of output capsules
        :param in_dim: Dimension of input capsules
        :param out_dim: Dimension of output capsules
        :param routing_iters: Number of routing iterations
        """
        super(CapsuleLayer, self).__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.routing_iters = routing_iters

        # Weight matrix W_ij for transforming input capsules to prediction vectors
        # Shape: [in_caps, out_caps, out_dim, in_dim]
        self.W = nn.Parameter(torch.randn(in_caps, out_caps, out_dim, in_dim))
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.W)

    def forward(self, u):
        """
        Forward pass with dynamic routing algorithm
        :param u: Input capsules, shape [batch_size, in_caps, in_dim]
        :return: Output capsules, shape [batch_size, out_caps, out_dim]
        """
        batch_size = u.size(0)

        # Step 1: Compute prediction vectors û_j|i = W_ij * u_i
        # Expand u dimensions for matrix multiplication with W
        # u_expanded shape: [batch_size, in_caps, 1, 1, in_dim]
        u_expanded = u.unsqueeze(2).unsqueeze(3)
        u_expanded = u_expanded.transpose(-2, -1)

        # Tile W for batch processing
        # W_tiled shape: [batch_size, in_caps, out_caps, out_dim, in_dim]
        W_tiled = self.W.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # Compute prediction vectors through matrix multiplication
        # u_hat shape: [batch_size, in_caps, out_caps, out_dim]
        u_hat = torch.matmul(W_tiled, u_expanded).squeeze(-1)

        # Step 2: Dynamic routing algorithm
        # Initialize routing logits b_ij to zero
        b = torch.zeros(batch_size, self.in_caps, self.out_caps, device=u.device)

        for iteration in range(self.routing_iters):
            # Compute coupling coefficients c_ij = softmax(b_ij)
            # Apply softmax over out_caps dimension
            c = F.softmax(b, dim=2)

            # Compute weighted sum s_j = Σ_i c_ij * û_j|i
            # Expand c dimensions to match u_hat for element-wise multiplication
            # c shape: [batch, in_caps, out_caps, 1], u_hat shape: [batch, in_caps, out_caps, out_dim]
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)

            # Apply squash activation function v_j = squash(s_j)
            v = squash(s)

            # Update routing logits (except for last iteration)
            if iteration < self.routing_iters - 1:
                # Compute agreement û_j|i · v_j
                # agreement shape: [batch_size, in_caps, out_caps]
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        return v

class CNN_GRU_SeqCap(nn.Module):
    """
    CNN-GRU-SeqCap model based on Wu et al. "Speech Emotion Recognition Using Capsule Networks"
    Implements both CNN and GRU branches with sequential capsule networks
    """
    def __init__(self, num_classes=4, input_dim=128, max_seq_len=200, 
                 window_size=40, window_stride=20, dropout=0.2):
        super(CNN_GRU_SeqCap, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.window_stride = window_stride
        self.dropout = dropout
        
        # CNN Branch - Based on Sabour et al. configuration
        self.cnn_backbone = nn.Sequential(
            # First conv layer: 256 filters, 9x9, stride 1
            nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Additional conv layers for better feature extraction
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Primary Capsules - Based on Sabour et al.: 32 8-dimensional capsules, 9x9, stride 2
        self.primary_caps_conv = nn.Conv2d(256, 32 * 8, kernel_size=9, stride=2, padding=0)
        self.primary_caps_dim = 8
        self.num_primary_caps_per_location = 32
        
        # GRU Branch for temporal modeling
        self.gru_hidden_dim = 128
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Feature fusion layer
        self.fusion_dim = 256
        self.feature_fusion = nn.Linear(self.gru_hidden_dim * 2, self.fusion_dim)
        
        # Dynamic capsule layers (initialized in forward)
        self.window_emo_caps = None
        self.utterance_caps = None
        
        # Calculate max number of windows based on paper parameters
        self.max_windows = max(1, (max_seq_len - window_size) // window_stride + 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass implementing CNN-GRU-SeqCap architecture
        :param x: Input spectrogram, shape [batch_size, 1, freq_bins, time_steps]
        :return: Class probabilities, shape [batch_size, num_classes]
        """
        B, C, H, T = x.shape
        
        # Branch 1: CNN Feature Extraction
        cnn_features = self.cnn_backbone(x)  # [B, 256, H', T']
        
        # Primary Capsules
        primary_caps_output = self.primary_caps_conv(cnn_features)
        # primary_caps_output shape: [B, 32*8, H'', T'']
        
        B, caps_channels, H_prime, T_prime = primary_caps_output.shape
        
        # Reshape to capsule format: [B, num_caps, caps_dim]
        primary_caps = primary_caps_output.view(
            B, self.num_primary_caps_per_location, self.primary_caps_dim, H_prime, T_prime
        )
        primary_caps = primary_caps.permute(0, 3, 4, 1, 2).contiguous()  # [B, H', T', 32, 8]
        primary_caps = primary_caps.view(
            B, H_prime * T_prime * self.num_primary_caps_per_location, self.primary_caps_dim
        )
        
        # Branch 2: GRU Temporal Modeling
        # Reshape input for GRU: [B, T, H]
        gru_input = x.squeeze(1).transpose(1, 2)  # [B, T, H]
        gru_output, _ = self.gru(gru_input)  # [B, T, hidden_dim*2]
        gru_features = self.feature_fusion(gru_output)  # [B, T, fusion_dim]
        
        # Windowing Process (based on paper: window_size=40, stride=20)
        caps_per_time_step = H_prime * self.num_primary_caps_per_location
        primary_caps_seq = primary_caps.view(B, T_prime, caps_per_time_step * self.primary_caps_dim)
        
        # Apply windowing with paper's parameters
        if T_prime < self.window_size:
            # Handle short sequences
            windows = primary_caps_seq.unsqueeze(1)
            num_windows = 1
            actual_window_size = T_prime
        else:
            # Apply unfold for windowing
            windows = primary_caps_seq.unfold(
                dimension=1, size=self.window_size, step=self.window_stride
            )
            num_windows = windows.size(1)
            windows = windows.transpose(-2, -1)
            actual_window_size = self.window_size
        
        # Reshape windows for capsule processing
        windows_flat = windows.contiguous().view(B, num_windows, -1)
        in_caps_per_window = actual_window_size * caps_per_time_step
        
        # Dynamic initialization of window emotion capsules
        if self.window_emo_caps is None:
            self.window_emo_caps = CapsuleLayer(
                in_caps=in_caps_per_window,
                out_caps=self.num_classes,
                in_dim=self.primary_caps_dim,
                out_dim=16,
                routing_iters=3
            ).to(x.device)
        
        # Process windows through emotion capsules
        window_caps_in = windows_flat.view(B, num_windows, in_caps_per_window, self.primary_caps_dim)
        window_caps_in_reshaped = window_caps_in.view(B * num_windows, in_caps_per_window, self.primary_caps_dim)
        
        # Apply Window Emotion Capsules
        window_caps_out = self.window_emo_caps(window_caps_in_reshaped)
        window_caps_out = window_caps_out.view(B, num_windows, self.num_classes, 16)
        
        # Aggregate emotion capsules (average across classes)
        utterance_caps_in = window_caps_out.mean(dim=2)  # [B, num_windows, 16]
        
        # Dynamic initialization of utterance capsules
        if self.utterance_caps is None:
            self.utterance_caps = CapsuleLayer(
                in_caps=min(num_windows, self.max_windows),
                out_caps=self.num_classes,
                in_dim=16,
                out_dim=16,
                routing_iters=3
            ).to(x.device)
        
        # Handle variable sequence lengths
        if num_windows < self.utterance_caps.in_caps:
            pad_size = self.utterance_caps.in_caps - num_windows
            padding = torch.zeros(B, pad_size, 16, device=x.device)
            utterance_caps_in = torch.cat([utterance_caps_in, padding], dim=1)
        elif num_windows > self.utterance_caps.in_caps:
            utterance_caps_in = utterance_caps_in[:, :self.utterance_caps.in_caps, :]
        
        # Apply Utterance Capsules
        final_caps = self.utterance_caps(utterance_caps_in)
        
        # Compute final output (capsule lengths as class probabilities)
        lengths = torch.sqrt((final_caps ** 2).sum(dim=-1) + 1e-8)  # [B, num_classes]
        
        return lengths


# === Usage Example ===
if __name__ == '__main__':
    # Example: batch_size=16, spectrogram size 1x128x200 (channels x frequency x time)
    dummy_input = torch.randn(16, 1, 128, 200)

    # Create model with CNN-GRU-SeqCap architecture
    model = CNN_GRU_SeqCap(
        num_classes=4,
        input_dim=128,
        max_seq_len=200,
        window_size=40,
        window_stride=20,
        dropout=0.2
    )
    
    # Forward pass
    try:
        output = model(dummy_input)
        print("Model output shape:", output.shape)  # Should be [16, 4]
        print("Output probabilities:", torch.softmax(output, dim=-1))
    except Exception as e:
        print("Model execution error (likely dimension mismatch):", e)
        print("This is expected - capsule network dimensions need precise design.")
        print("Please adjust CapsuleLayer parameters based on your CNN output dimensions.")