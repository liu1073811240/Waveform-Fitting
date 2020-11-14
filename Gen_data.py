import torch

# 在100到200范围内生成5个数据
train_data = torch.randint(100, 200, (5, ), dtype=torch.float32)

for i in range(5):
    # 在生成的五个数据中让它们有一定范围性的波动（-2，2）
    gen_data = train_data + torch.randint(-2, 2, (len(train_data), ))
    # print(train_data)
    # print(gen_data)
    train_data = torch.cat([train_data, gen_data])  # 以行进行拼接数据

print(train_data)
print(len(train_data))  # 160个数据
test_data = train_data + torch.randint(-2, 2, (len(train_data), ))
print(test_data)
torch.save(train_data, "./train.data")
torch.save(test_data, "./test.data")



