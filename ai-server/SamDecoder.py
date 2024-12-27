
from modeling import ImageEncoderViT
import torch
from torch import nn
from torch.nn import functional as F
last_layer_no = 344
class SAM_Decoder(nn.Module):
    def __init__(
        self,
        sam_encoder:ImageEncoderViT,
        num_classes:int=9
    ) -> None:
    
        super().__init__()
        self.sam_encoder = sam_encoder(img_size=264)
        for layer_no, param in enumerate(self.sam_encoder.parameters()):
            if(layer_no > (last_layer_no - 6)):
                param.requires_grad = True
            else:
                param.requires_grad = False  
        self.classification_Head =MLP(256,num_classes)
        
    def forward(
        self,x
    ) -> torch.Tensor:
        x = self.sam_encoder(x)
        # print(x.shape)
        iou_pred = self.classification_Head(x)

        return iou_pred

    # def predict(
    #     self,
    #     image_embeddings: torch.Tensor,
    #     image_pe: torch.Tensor,
    #     sparse_prompt_embeddings: torch.Tensor,
    #     dense_prompt_embeddings: torch.Tensor,
    # ) ->  torch.Tensor:
    #     """Predicts masks. See 'forward' for more details."""
    #     # Concatenate output tokens
    #     output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0).to(DEVICE)
    #     output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    #     tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    #     # Expand per-image data in batch direction to be per-mask
    #     src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    #     src = src + dense_prompt_embeddings
    #     pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    #     b, c, h, w = src.shape

    #     # Run the transformer
    #     hs, src = self.transformer(src, pos_src, tokens)
    #     src = src.transpose(1, 2).view(b, c, h, w)
    #     upscaled_embedding = self.output_upscaling(src)
    #     # print(upscaled_embedding.shape)
    #     b, c, h, w = upscaled_embedding.shape
    #     # print(src.shape) 
    #     # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :
        
    #     iou_pred = self.classification_Head(upscaled_embedding)

    #     return iou_pred


class MLP(nn.Module):
    def __init__(
        self,
        channels: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(256, 1024)
        self.batch_norm1=nn.BatchNorm1d(1024,momentum=0.3)
        self.dropout1=nn.Dropout(p=0.2)
        
        self.linear2 = nn.Linear(1024, 512)
        self.batch_norm2=nn.BatchNorm1d(512,momentum=0.3)
        self.dropout2=nn.Dropout(p=0.2)

        self.linear3 =nn.Linear(512,256)
        self.batch_norm3=nn.BatchNorm1d(256,momentum=0.3)
        self.dropout3=nn.Dropout(p=0.25)
        
        self.linear4 =nn.Linear(256,128)
        self.batch_norm4=nn.BatchNorm1d(128,momentum=0.3)
        self.dropout4=nn.Dropout(p=0.3)

        self.linear5 =nn.Linear(128,64)
        self.batch_norm5=nn.BatchNorm1d(64,momentum=0.3)
        self.dropout5=nn.Dropout(p=0.4)
        
        self.linear_out=nn.Linear(64,out_features)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.global_avg_pool(x)
        x = x.view(b, c)
    
        x = self.linear1(x)
        x = self.batch_norm1(x)  # Batch normalization
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.batch_norm2(x)  # Batch normalization
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = self.batch_norm3(x)  # Batch normalization
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.linear4(x)
        x = self.batch_norm4(x)  # Batch normalization
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.linear5(x)
        x = self.batch_norm5(x)  # Batch normalization# Batch normalization
        x = F.relu(x)
        x = self.dropout5(x)
        
        x = self.linear_out(x)
        x = F.softmax(x, dim=1)
        return x