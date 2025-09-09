import torch
import torch.nn.functional as F

import time

# CNN Model 
#         Parameters: 896450 
#         runtime: 0.0228 seconds
# ViT Model 
#         Parameters: 17430914 
#         runtime: 0.0634 seconds
# Hybrid Model 
#         Parameters: 3798146 
#         runtime: 0.0189 seconds

#################################
# attention experiments: num_layers
#################################
# Segmentation Model 4 layers att
#         Parameters: 8722145 
#         runtime: 0.0282 seconds
#         final accuracy: 95.68%
# Segmentation Model 3 layers att
#         Parameters: 7407073 
#         runtime: 0.0260 seconds
#         final accuracy: 95.84%
# Segmentation Model 2 layers att
#         Parameters: 6092001 
#         runtime: 0.0216 seconds
#         final accuracy: 95.26%

#################################
# attention experiments: patch_size
#################################
# I think a smaller patch size and increased dataset will have a substantial effect on results.
#  - With patch size 8, there is noticable patching on output
#  - I am achieving 95% which is considerably good (best results out of 3 att layers), but i want tighter segmentations.

# Segmentation Model 4 layers att, patch size 4 (4 times more attention paramt)
#         Parameters: 7195361 
#         runtime: 0.0654 seconds
#         final accuracy: 95.09%


def compare_models(cfg_seg):
    model_cnn = CNNModel()
    model_vit = ViTModel()
    model_hybrid = HybridModel()
    model_seg = ClassSegmentationModel(cfg_seg)

    total_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    total_params_vit = sum(p.numel() for p in model_vit.parameters())
    total_params_hybrid = sum(p.numel() for p in model_hybrid.parameters())
    total_params_seg = sum(p.numel() for p in model_seg.parameters())

    # make fake torch input (3,256,256)
    fake_input = torch.randn(1, 3, 256, 256)

    start = time.time()
    _ = model_cnn(fake_input)
    runtime_cnn = time.time() - start

    start = time.time()
    _ = model_vit(fake_input)
    runtime_vit = time.time() - start

    start = time.time()
    _ = model_hybrid(fake_input)
    runtime_hybrid = time.time() - start

    start = time.time()
    _ = model_seg(fake_input)
    runtime_seg = time.time() - start

    print(f"CNN Model \n\tParameters: {total_params_cnn} \n\truntime: {runtime_cnn:.4f} seconds")
    print(f"ViT Model \n\tParameters: {total_params_vit} \n\truntime: {runtime_vit:.4f} seconds")
    print(f"Hybrid Model \n\tParameters: {total_params_hybrid} \n\truntime: {runtime_hybrid:.4f} seconds")
    print(f"Segmentation Model \n\tParameters: {total_params_seg} \n\truntime: {runtime_seg:.4f} seconds")

#################################
# Standard Models
#################################

class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = torch.nn.Linear(128 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        # conv1
        # 256 x 256 x 3 -> 128 x 128 x 32
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # conv2
        # 128 x 128 x 32 -> 64 x 64 x 64
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # conv3
        # 64 x 64 x 64 -> 32 x 32 x 128
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = self.adaptive_pool(x)

        # flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ViTModel(torch.nn.Module):
    def __init__(self, input_shape=(3, 256, 256), patch_size=16, num_classes=2, num_dims=768, num_heads=8, num_layers=3):
        super(ViTModel, self).__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.num_patches = (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
        
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.num_dims))
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.num_dims) 
        )

        #############
        # Layers
        #############

        self.patch_embedding = torch.nn.Linear(
            3 * patch_size * patch_size, 
            self.num_dims
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.num_dims, 
            nhead=self.num_heads, 
            batch_first=True,
            dropout=0.1  # Add dropout for regularization
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )

        self.norm = torch.nn.LayerNorm(self.num_dims)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.num_dims, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)

        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.positional_embedding

        x = self.transformer(x)
        
        x = self.norm(x)
        
        cls_output = x[:, 0] 
        
        output = self.classifier(cls_output)

        return output

class HybridModel(torch.nn.Module):
    def __init__(self, input_shape=(3, 256, 256), patch_size=8, num_classes=2, num_dims=256, num_heads=8, num_layers=1):

        super(HybridModel, self).__init__()

        self.num_dims = num_dims
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.cnn_out_channels = 64
        self.patches = input_shape[1] // (4*self.patch_size) * input_shape[2] // (4*self.patch_size)

        # CNN
        self.cnn = torch.nn.Sequential(
            # (3,256,256) -> (32,128,128) 
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),

            # (32,128,128) -> (64,64,64)
            torch.nn.Conv2d(32,  self.cnn_out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )

        # projection to transformer dimension
        self.proj = torch.nn.Linear(self.cnn_out_channels*patch_size*patch_size, num_dims)


        self.cls = torch.nn.Parameter(torch.randn(1, 1, num_dims))
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, self.patches + 1, num_dims)
        )

        # transformer for attention
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=num_dims, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # normalization
        self.norm = torch.nn.LayerNorm(num_dims)

        # classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(num_dims, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, _, _, _ = x.shape
        

        x = self.cnn(x) # -> (B, 64, 64, 64)

        # (B, 64, 64, 64) -> (B, 64, image_patches[1] // patch_size, image_patches[2] // patch_size, patch_size, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, 64, 8, 8, 8, 8) -> (B, 8, 8, 64, 8, 8)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, 8, 8, 64, 8, 8) -> (B, 64, 64*8*8)
        x = x.view(B, -1,  self.cnn_out_channels * self.patch_size * self.patch_size)

        # project to num_dims
        x = self.proj(x)

        cls_tokens = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.positional_embedding

        x = self.transformer(x)
        
        x = self.norm(x)
        
        cls_output = x[:, 0] 
        
        output = self.classifier(cls_output)

        return output

#################################
# Segmentation Models
#################################

class InitialClassSegmentationModel(torch.nn.Module):
    def __init__(self, cfg):
        input_shape = cfg.get("input_shape", (3, 256, 256))
        patch_size = cfg.get("patch_size", 8)
        num_classes = cfg.get("num_classes", 2)
        num_dims = cfg.get("num_dims", 256)
        num_heads = cfg.get("num_heads", 8)
        num_layers = cfg.get("num_layers", 4)

        super(ClassSegmentationModel, self).__init__()

        self.num_dims = num_dims
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.patches = 64
        self.cnn_out_channels = 64
        self.h = input_shape[1] // (4*self.patch_size)
        self.w = input_shape[2] // (4*self.patch_size)
        self.patches = self.h * self.w

        # CNN
        self.conv1 = torch.nn.Sequential(
            # (3,256,256) -> (32,128,128) 
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            # (32,128,128) -> (64,64,64)
            torch.nn.Conv2d(32,  self.cnn_out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )

        # projection to transformer dimension
        self.proj = torch.nn.Linear(self.cnn_out_channels*patch_size*patch_size, num_dims)
        self.inv_proj = torch.nn.Linear(num_dims, self.cnn_out_channels*patch_size*patch_size)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, num_dims))
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, self.patches + 1, num_dims)
        )

        # transformer for attention
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=num_dims, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # normalization
        self.norm = torch.nn.LayerNorm(num_dims)

        # upsampling head with skip connections

        # (B, 64, 64, 64) -> (B, 32, 128, 128)
        self.upsample1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.cnn_out_channels, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        # (B, 64, 128, 128) -> (B, 1, 256, 256)
        self.upsample2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
        )

        
    def forward(self, x):
        B, _, H, W = x.shape
        

        conv1 = self.conv1(x) # -> (B, 32, 128, 128)
        conv2 = self.conv2(conv1) # -> (B, 64, 64, 64)

        # (B, 64, 64, 64) -> (B, 64, image_patches[1] // patch_size, image_patches[2] // patch_size, patch_size, patch_size)
        x = conv2.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, 64, 8, 8, 8, 8) -> (B, 8, 8, 64, 8, 8)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, 8, 8, 64, 8, 8) -> (B, 64, 64*8*8)
        x = x.view(B, -1,  self.cnn_out_channels * self.patch_size * self.patch_size)

        # project to num_dims
        x = self.proj(x)

        cls_tokens = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.positional_embedding

        x = self.transformer(x)
        
        x = self.norm(x)
        
        #cls_output = x[:, 0] 
        image_tokens = x[:, 1:]

        # project to patches
        image_tokens = self.inv_proj(image_tokens)

        # reshape image tokens back to image grid
        # (B, 64, 64*8*8) -> (B, 8, 8, 64, 8, 8)
        image_tokens = image_tokens.view(B, self.h, self.w, self.cnn_out_channels, self.patch_size, self.patch_size)
        # (B, 8, 8, 64, 8, 8) -> (B, 64, 8, 8, 8, 8)
        image_tokens = image_tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        # (B, 64, image_patches[1] // patch_size, image_patches[2] // patch_size, patch_size, patch_size) -> (B, 64, 64, 64)
        image_tokens = image_tokens.view(B, self.cnn_out_channels, self.h * self.patch_size, self.w * self.patch_size)
        
        # upsample back to original image size with skips
        x = self.upsample1(image_tokens)
        output = self.upsample2(torch.cat([x, conv1], dim=1))

        return output

class ClassSegmentationModel(torch.nn.Module):
    def __init__(self, cfg):
        input_shape = cfg.get("input_shape", (3, 512, 512))
        patch_size = cfg.get("patch_size", 8)
        num_classes = cfg.get("num_classes", 2)
        num_dims = cfg.get("num_dims", 256)
        num_heads = cfg.get("num_heads", 8)
        num_layers = cfg.get("num_layers", 4)
        self.num_convs = 3

        super(ClassSegmentationModel, self).__init__()

        self.num_dims = num_dims
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.patches = 64
        self.cnn_out_channels = 128
        self.h = input_shape[1] // ((2**self.num_convs)*self.patch_size)
        self.w = input_shape[2] // ((2**self.num_convs)*self.patch_size)
        self.patches = self.h * self.w

        # CNN
        self.conv1 = torch.nn.Sequential(
            # (3,512,512) -> (32,256,256) 
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            # (32,256,256) -> (64,128,128)
            torch.nn.Conv2d(32,  64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )

        self.conv3 = torch.nn.Sequential(
            # (64,128,128) -> (128,64,64)
            torch.nn.Conv2d(64,  self.cnn_out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )

        # projection to transformer dimension
        self.proj = torch.nn.Linear(self.cnn_out_channels*patch_size*patch_size, num_dims)
        self.inv_proj = torch.nn.Linear(num_dims, self.cnn_out_channels*patch_size*patch_size)

        self.cls = torch.nn.Parameter(torch.randn(1, 1, num_dims))
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, self.patches + 1, num_dims)
        )

        # transformer for attention
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=num_dims, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # normalization
        self.norm = torch.nn.LayerNorm(num_dims)

        # upsampling head with skip connections

        # (B, 128, 64, 64) -> (B, 64, 128, 128)
        self.upsample1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.cnn_out_channels, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        # (B, 64, 128, 128) -> (B, 32, 256, 256)
        self.upsample2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        # (B, 32, 256, 256) -> (B, 1, 512, 512)
        self.upsample3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
        )

        
    def forward(self, x):
        B, _, H, W = x.shape
        

        conv1 = self.conv1(x) # -> (B, 32, 256, 256)
        conv2 = self.conv2(conv1) # -> (B, 64, 128, 128)
        conv3 = self.conv3(conv2) # -> (B, 128, 64, 64)

        # (B, 64, 64, 64) -> (B, 64, image_patches[1] // patch_size, image_patches[2] // patch_size, patch_size, patch_size)
        x = conv3.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (B, 64, 8, 8, 8, 8) -> (B, 8, 8, 64, 8, 8)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, 8, 8, 64, 8, 8) -> (B, 64, 64*8*8)
        x = x.view(B, -1,  self.cnn_out_channels * self.patch_size * self.patch_size)

        # project to num_dims
        x = self.proj(x)

        cls_tokens = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.positional_embedding

        x = self.transformer(x)
        
        x = self.norm(x)
        
        #cls_output = x[:, 0] 
        image_tokens = x[:, 1:]

        # project to patches
        image_tokens = self.inv_proj(image_tokens)

        # reshape image tokens back to image grid
        # (B, 64, 64*8*8) -> (B, 8, 8, 64, 8, 8)
        image_tokens = image_tokens.view(B, self.h, self.w, self.cnn_out_channels, self.patch_size, self.patch_size)
        # (B, 8, 8, 64, 8, 8) -> (B, 64, 8, 8, 8, 8)
        image_tokens = image_tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        # (B, 64, image_patches[1] // patch_size, image_patches[2] // patch_size, patch_size, patch_size) -> (B, 64, 64, 64)
        image_tokens = image_tokens.view(B, self.cnn_out_channels, self.h * self.patch_size, self.w * self.patch_size)
        
        # upsample back to original image size with skips
        x = self.upsample1(image_tokens)
        x = self.upsample2(torch.cat([x, conv2], dim=1))
        output = self.upsample3(torch.cat([x, conv1], dim=1))

        return output
