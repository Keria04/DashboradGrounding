"""
# -*- coding: utf-8 -*-
日志工具模块
用于配置和管理项目日志
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(
    name: str = "dashboard_grounding",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days"
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
        rotation: 日志轮转频率
        retention: 日志保留时间
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 设置日志格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler进行日志轮转
        if rotation == "1 day":
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8'
            )
        else:
            # 使用大小轮转
            max_bytes = 10 * 1024 * 1024  # 10MB
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=5,
                encoding='utf-8'
            )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 防止日志重复
    logger.propagate = False
    
    return logger


class Logger:
    """自定义日志类"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
        """
        self.logger = setup_logger(name, log_file=log_file)
        self.start_time = datetime.datetime.now()
    
    def info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录严重错误日志"""
        self.logger.critical(message)
    
    def log_training_start(self, config: dict):
        """记录训练开始"""
        self.info("=" * 50)
        self.info("开始训练")
        self.info(f"开始时间: {self.start_time}")
        self.info(f"配置: {config}")
        self.info("=" * 50)
    
    def log_training_end(self, metrics: dict):
        """记录训练结束"""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        
        self.info("=" * 50)
        self.info("训练完成")
        self.info(f"结束时间: {end_time}")
        self.info(f"训练时长: {duration}")
        self.info(f"最终指标: {metrics}")
        self.info("=" * 50)
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """记录epoch信息"""
        self.info(f"Epoch {epoch}:")
        self.info(f"  训练指标: {train_metrics}")
        self.info(f"  验证指标: {val_metrics}")
    
    def log_experiment_info(self, experiment_dir: str, config: dict):
        """记录实验信息"""
        self.info(f"实验目录: {experiment_dir}")
        self.info(f"实验配置: {config}")


def get_logger(name: str = "dashboard_grounding") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)


def log_model_info(logger: logging.Logger, model, input_shape: tuple):
    """
    记录模型信息
    
    Args:
        logger: 日志记录器
        model: 模型实例
        input_shape: 输入形状
    """
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型参数统计:")
    logger.info(f"  总参数数: {total_params:,}")
    logger.info(f"  可训练参数数: {trainable_params:,}")
    logger.info(f"  输入形状: {input_shape}")
    
    # 记录模型结构（简化版）
    if hasattr(model, 'model'):
        logger.info(f"  模型类型: {type(model.model).__name__}")
    else:
        logger.info(f"  模型类型: {type(model).__name__}")


def log_data_info(logger: logging.Logger, data_stats: dict):
    """
    记录数据信息
    
    Args:
        logger: 日志记录器
        data_stats: 数据统计信息
    """
    logger.info("数据集统计:")
    for key, value in data_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    batch_idx: int,
    total_batches: int,
    loss: float,
    metrics: dict = None
):
    """
    记录训练进度
    
    Args:
        logger: 日志记录器
        epoch: 当前epoch
        batch_idx: 当前批次索引
        total_batches: 总批次数
        loss: 当前损失
        metrics: 其他指标
    """
    progress = (batch_idx + 1) / total_batches * 100
    message = f"Epoch {epoch}, Batch {batch_idx+1}/{total_batches} ({progress:.1f}%), Loss: {loss:.4f}"
    
    if metrics:
        for key, value in metrics.items():
            message += f", {key}: {value:.4f}"
    
    logger.info(message)


if __name__ == "__main__":
    # 测试日志功能
    logger = setup_logger("test_logger", log_file="logs/test.log")
    
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    # 测试自定义日志类
    custom_logger = Logger("custom_test", "logs/custom_test.log")
    custom_logger.info("自定义日志测试")
    
    print("日志测试完成！")
