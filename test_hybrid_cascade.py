"""
HybridCascade-LaFTer训练器测试脚本
验证实现是否能正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from dassl.config import get_cfg_default
from trainers.HybridCascadeLaFTer import HybridCascadeLaFTerUFT, load_hybrid_cascade_clip_to_cpu
from utils.data_utils import ds_specific_templates


def test_hybrid_cascade_clip_loading():
    """测试HybridCascade CLIP模型加载"""
    print("=== Testing HybridCascade CLIP Model Loading ===")
    
    # 创建配置
    cfg = get_cfg_default()
    cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
    
    try:
        # 加载模型
        clip_model = load_hybrid_cascade_clip_to_cpu(cfg)
        print(f"Successfully loaded HybridCascade CLIP model: {type(clip_model)}")
        print(f"Visual model type: {type(clip_model.visual)}")
        
        # 检查关键方法是否存在
        assert hasattr(clip_model.visual, 'forward_with_intermediates'), "Missing forward_with_intermediates method"
        assert hasattr(clip_model.visual, 'embeddings_patch'), "Missing embeddings_patch method"
        assert hasattr(clip_model.visual, 'extraction_indices'), "Missing extraction_indices attribute"
        assert hasattr(clip_model.visual, 'stage_boundaries'), "Missing stage_boundaries attribute"
        
        print(f"extraction_indices: {clip_model.visual.extraction_indices}")
        print(f"stage_boundaries: {clip_model.visual.stage_boundaries}")
        
        return clip_model
        
    except Exception as e:
        print(f"Failed to load HybridCascade CLIP model: {e}")
        return None


def test_feature_extraction():
    """测试特征提取功能"""
    print("\n=== Testing Feature Extraction ===")
    
    clip_model = test_hybrid_cascade_clip_loading()
    if clip_model is None:
        return False
    
    try:
        # 创建测试输入
        batch_size = 2
        test_image = torch.randn(batch_size, 3, 224, 224)
        
        # 测试标准前向传播
        with torch.no_grad():
            standard_features = clip_model.visual(test_image)
            print(f"Standard forward pass successful, output shape: {standard_features.shape}")
            
        # 测试patch embeddings
        with torch.no_grad():
            patch_features = clip_model.visual.embeddings_patch(test_image)
            print(f"Patch embeddings successful, output shape: {patch_features.shape}")
            
        # 测试中间特征提取（从patch embeddings开始）
        with torch.no_grad():
            final_features, intermediate_features = clip_model.visual.forward_with_intermediates(patch_features)
            print(f"Intermediate feature extraction successful")
            print(f"   - Final feature shape: {final_features.shape}")
            print(f"   - Number of intermediate features: {len(intermediate_features)}")
            for i, feat in enumerate(intermediate_features):
                print(f"   - Intermediate feature {i} shape: {feat.shape}")
                
        return True
        
    except Exception as e:
        print(f"Feature extraction test failed: {e}")
        return False


def test_hybrid_cascade_model():
    """测试HybridCascadeLaFTerUFT模型"""
    print("\n=== Testing HybridCascadeLaFTerUFT Model ===")
    
    clip_model = test_hybrid_cascade_clip_loading()
    if clip_model is None:
        return False
    
    try:
        # 模拟类名和模板
        classnames = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        templates = ['a photo of a {}']
        ds_templates = ds_specific_templates.get('Emotion6', ['a photo of a {}'])
        
        # 创建HybridCascade模型
        from trainers.HybridCascadeLaFTer import HybridCascadeLaFTerUFT
        
        model = HybridCascadeLaFTerUFT(
            model=clip_model,
            classes=classnames,
            templates=templates,
            ds_templates=ds_templates,
            dataset_name='Emotion6',
            txt_cls='lafter',
            device='cpu'
        )
        
        print(f"Successfully created HybridCascadeLaFTerUFT model")
        print(f"   - Number of stages: {len(model.stage_boundaries)}")
        print(f"   - Number of NGA aggregators: {len(model.nga_layers)}")
        print(f"   - Extraction layer indices: {model.extraction_indices}")
        
        # 测试前向传播
        batch_size = 2
        test_image = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model.forward_supervised(test_image)
            print(f"Forward pass successful, output shape: {output.shape}")
            
            # 测试带一致性损失的前向传播
            output_with_loss, consistency_loss = model.forward(test_image, torch.tensor([0, 1]))
            print(f"Forward pass with loss successful")
            print(f"   - Output shape: {output_with_loss.shape}")
            print(f"   - Consistency loss: {consistency_loss.item():.4f}")
            
        return True
        
    except Exception as e:
        print(f"HybridCascade model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("Starting HybridCascade-LaFTer tests\n")
    
    tests = [
        test_hybrid_cascade_clip_loading,
        test_feature_extraction,
        test_hybrid_cascade_model
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("All tests passed! HybridCascade-LaFTer implementation works correctly")
    else:
        print("Some tests failed, please check implementation")


if __name__ == "__main__":
    main()