#!/usr/bin/env python3
"""
测试脚本：验证MultiLayerLaFTer训练器的基本功能
阶段2：dassl集成测试
"""

import torch
import argparse
from LaFTer import setup_cfg
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

def test_trainer_registration():
    """测试训练器注册"""
    print("=== 测试阶段2：dassl集成测试 ===")
    
    # 创建基本参数
    args = argparse.Namespace()
    args.trainer = "MultiLayerLaFTer"  # 使用我们的多层prompt训练器
    args.config_file = ""
    args.dataset_config_file = ""
    args.root = "data"
    args.output_dir = "output/test"
    args.resume = ""
    args.seed = 1
    args.source_domains = None
    args.target_domains = None
    args.transforms = None
    args.backbone = ""
    args.head = ""
    args.txt_cls = "cls_only"  # 简单模式
    args.arch = "ViT-B/32"
    args.multi_layer_prompt = True  # 启用多层prompt
    args.opts = []
    
    try:
        print("1. 设置配置...")
        cfg = setup_cfg(args)
        
        print("2. 设置随机种子...")
        if cfg.SEED >= 0:
            set_random_seed(cfg.SEED)
        
        print("3. 设置日志...")
        setup_logger(cfg.OUTPUT_DIR)
        
        print("4. 尝试构建训练器...")
        trainer = build_trainer(cfg)
        print(f"✅ 训练器构建成功: {type(trainer).__name__}")
        
        print("5. 检查模型...")
        model = trainer.model
        print(f"✅ 模型创建成功: {type(model).__name__}")
        
        print("6. 检查数据加载器...")
        if hasattr(trainer, 'train_loader_x') and trainer.train_loader_x is not None:
            print(f"✅ 训练数据加载器存在")
        else:
            print("⚠️  训练数据加载器不存在")
            
        print("7. 检查优化器...")
        if hasattr(trainer, '_optims') and trainer._optims:
            print(f"✅ 优化器设置成功: {list(trainer._optims.keys())}")
        else:
            print("⚠️  优化器未设置")
            
        return True, trainer
        
    except Exception as e:
        print(f"❌ 训练器构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_single_forward_pass(trainer):
    """测试单个batch的前向传播"""
    print("\n=== 测试单batch前向传播 ===")
    
    try:
        # 创建虚拟数据
        batch_size = 2
        img_size = 224
        num_classes = 6
        
        dummy_batch = {
            'img': torch.randn(batch_size, 3, img_size, img_size).cuda(),
            'label': torch.randint(0, num_classes, (batch_size,)).cuda()
        }
        
        print("1. 创建虚拟batch...")
        print(f"   - 图像形状: {dummy_batch['img'].shape}")
        print(f"   - 标签形状: {dummy_batch['label'].shape}")
        
        print("2. 测试模型推理...")
        trainer.model.eval()
        with torch.no_grad():
            logits = trainer.model_inference(dummy_batch['img'])
            print(f"✅ 推理成功，输出形状: {logits.shape}")
        
        print("3. 测试前向传播...")
        trainer.model.train()
        loss_summary = trainer.forward_backward(dummy_batch)
        print(f"✅ 前向传播成功")
        print(f"   - 损失: {loss_summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始MultiLayerLaFTer训练器测试...")
    
    # 测试1: 训练器注册和初始化
    success, trainer = test_trainer_registration()
    
    if not success:
        print("\n❌ 阶段2测试失败：训练器无法正确注册/初始化")
        return False
    
    # 测试2: 单batch前向传播
    if trainer is not None:
        forward_success = test_single_forward_pass(trainer)
        
        if forward_success:
            print("\n✅ 阶段2测试成功：dassl集成正常工作")
            return True
        else:
            print("\n⚠️  阶段2部分成功：训练器初始化正常，但前向传播有问题")
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 MultiLayerLaFTer训练器已准备好进入阶段3（完整训练测试）")
    else:
        print("\n🔧 需要进一步修复才能进入下一阶段")