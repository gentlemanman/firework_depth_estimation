predict.py: 给定包含rgb图像的文件夹和训练好的模型model，进行预测。
	输出：dpt(可视化深度图)、dpt_cv(用于重建的深度图)、rgb_480x480(将原始的rgb图像change size)
ground_truth:如果存在该文件夹，那就是原始深度图。data_process.py可以将其处理成合适的大小。
predict_backup:用来备份之前的预测结果