# -*- coding: utf-8 -*-
"""
MNIST数字识别完整优化实验代码
包含网络结构调整、优化算法选择、批量大小和迭代次数调整的实验
"""

from keras import datasets, utils, Sequential, layers, optimizers, callbacks
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
import pandas as pd
import os
from PIL import Image

# 设置随机种子保证可重复性
np.random.seed(42)
tf.random.set_seed(42)


# 数据加载与预处理
def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # 归一化并转换数据类型
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot编码标签
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def preprocess_external_image(image_path):
    """处理外部输入的图片"""
    # 打开图片并转换为灰度
    img = Image.open(image_path).convert('L')
    # 调整大小为28x28（MNIST标准尺寸）
    img = img.resize((28, 28))
    # 转换为numpy数组
    img_array = np.array(img)
    # 反转颜色（MNIST是白底黑字，如果输入是黑底白字需要反转）
    img_array = 255 - img_array
    # 归一化
    img_array = img_array.astype('float32') / 255.0
    # 添加batch维度 -> (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# 定义不同的网络结构
def build_model(model_type='default'):
    if model_type == 'default':
        model = Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),

            layers.Dense(10, activation='softmax')
        ])
    elif model_type == 'deep':
        model = Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),

            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(10, activation='softmax')
        ])
    elif model_type == 'wide':
        model = Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),

            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            layers.Dense(10, activation='softmax')
        ])
    elif model_type == 'simple':
        model = Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),

            layers.Dense(10, activation='softmax')
        ])
    else:
        raise ValueError("Unknown model type")

    return model


# 训练配置与回调函数
def get_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]


# 可视化训练过程
def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{title} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{title} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


def recognize_custom_image(model):
    """识别用户提供的图片"""
    while True:
        print("\n--- 图片识别功能 ---")
        print("1. 识别新图片")
        print("2. 返回主菜单")
        choice = input("请选择操作: ")

        if choice == '2':
            break

        image_path = input("请输入图片路径: ").strip('"')  # 处理可能有的引号

        if not os.path.exists(image_path):
            print("错误：文件不存在")
            continue

        try:
            # 预处理图片
            processed_img = preprocess_external_image(image_path)

            # 预测
            pred_probs = model.predict(processed_img, verbose=0)
            pred_label = np.argmax(pred_probs)
            confidence = np.max(pred_probs)

            # 显示结果
            print(f"\n预测结果: {pred_label} (置信度: {confidence:.2%})")

            # 可视化
            plt.figure(figsize=(6, 6))
            plt.imshow(processed_img[0], cmap='gray')
            plt.title(f'Predicted: {pred_label}\nConfidence: {confidence:.2%}')
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"处理出错: {str(e)}")


# 运行单个实验
def run_experiment(model_type='default', optimizer_type='adam',
                   learning_rate=0.001, batch_size=256, epochs=50):
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()

    # 构建模型
    model = build_model(model_type)

    # 选择优化器
    if optimizer_type == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unknown optimizer type")

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 训练模型
    start_time = time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=get_callbacks(),
        verbose=0
    )
    training_time = time() - start_time

    # 加载最佳模型
    model = tf.keras.models.load_model('best_model.keras')

    # 评估测试集
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # 可视化训练过程
    title = f"Model: {model_type}, Optimizer: {optimizer_type}, LR: {learning_rate}, Batch: {batch_size}"
    plot_training_history(history, title)

    return {
        'model_type': model_type,
        'optimizer': optimizer_type,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': len(history.history['loss']),  # 实际运行的epoch数
        'training_time': round(training_time, 2),
        'train_acc': round(max(history.history['accuracy']), 4),
        'val_acc': round(max(history.history['val_accuracy']), 4),
        'test_acc': round(test_acc, 4),
        'test_loss': round(test_loss, 4)
    }


# 运行所有实验
def run_all_experiments():
    # 定义实验参数
    model_types = ['default', 'deep', 'wide', 'simple']
    optimizers = [
        ('adam', 0.001),
        ('adam', 0.0001),
        ('sgd', 0.01),
        ('sgd', 0.001),
        ('rmsprop', 0.001)
    ]
    batch_sizes = [32, 128, 256, 512]
    epochs = 50

    results = []

    # 运行网络结构实验
    print("=" * 50)
    print("Running Network Architecture Experiments")
    print("=" * 50)
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        result = run_experiment(
            model_type=model_type,
            optimizer_type='adam',
            learning_rate=0.001,
            batch_size=256,
            epochs=epochs
        )
        results.append(result)

    # 运行优化器实验 (使用最佳网络结构)
    print("\n" + "=" * 50)
    print("Running Optimizer Experiments")
    print("=" * 50)
    best_model_type = max(results, key=lambda x: x['test_acc'])['model_type']
    for opt, lr in optimizers:
        print(f"\nTraining with {opt} optimizer (lr={lr})...")
        result = run_experiment(
            model_type=best_model_type,
            optimizer_type=opt,
            learning_rate=lr,
            batch_size=256,
            epochs=epochs
        )
        results.append(result)

    # 运行批量大小实验 (使用最佳网络和优化器)
    print("\n" + "=" * 50)
    print("Running Batch Size Experiments")
    print("=" * 50)
    best_config = max(results, key=lambda x: x['test_acc'])
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size {batch_size}...")
        result = run_experiment(
            model_type=best_config['model_type'],
            optimizer_type=best_config['optimizer'],
            learning_rate=best_config['learning_rate'],
            batch_size=batch_size,
            epochs=epochs
        )
        results.append(result)

    # 保存结果到DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv('mnist_experiment_results.csv', index=False)

    # 打印最佳结果
    best_result = df_results.loc[df_results['test_acc'].idxmax()]
    print("\n" + "=" * 50)
    print("Best Experiment Results:")
    print("=" * 50)
    print(best_result)

    return df_results


# 主程序
if __name__ == "__main__":
    # 运行所有实验
    results_df = run_all_experiments()

    # 显示所有实验结果
    print("\nAll Experiment Results:")
    print(results_df.sort_values('test_acc', ascending=False).head(10))

    # 加载最佳模型用于交互式测试
    best_model = tf.keras.models.load_model('best_model.keras')
    (x_train, y_train), (x_test, y_test) = load_data()

    while True:
        print("\n--- 主菜单 ---")
        print("1. 查看测试集随机样本")
        print("2. 识别自定义图片")
        print("3. 显示实验结果")
        print("4. 退出")
        choice = input("请选择操作: ")

        if choice == '1':
            # 随机选择测试样本展示
            idx = np.random.randint(0, len(x_test))
            sample = x_test[idx]
            true_label = np.argmax(y_test[idx])

            # 预测
            pred = best_model.predict(np.expand_dims(sample, axis=0), verbose=0)
            pred_label = np.argmax(pred)
            confidence = np.max(pred)

            # 显示结果
            plt.figure(figsize=(6, 6))
            plt.imshow(sample, cmap='gray')
            plt.title(f'True: {true_label}, Predicted: {pred_label}\nConfidence: {confidence:.2%}')
            plt.axis('off')
            plt.show()

        elif choice == '2':
            recognize_custom_image(best_model)

        elif choice == '3':
            print("\n实验最佳结果:")
            best_result = results_df.loc[results_df['test_acc'].idxmax()]
            print(best_result)

            print("\nTop 10 模型:")
            print(results_df.sort_values('test_acc', ascending=False).head(10))

        elif choice == '4':
            print("退出程序...")
            break

        else:
            print("无效输入，请重新选择")