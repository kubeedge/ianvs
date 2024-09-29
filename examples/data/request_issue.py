import json
import requests

# 从 JSON 文件读取数据
with open('test_data.json', 'r') as f:
    data = json.load(f)

# 用于保存所有提取的数据
all_extracted_data = []

# 遍历每个条目
for item in data:
    title = item['title']
    comments_url = item['comments_url']

    # 从评论 URL 获取评论数据
    response = requests.get(comments_url)

    if response.status_code == 200:
        comments = response.json()

        # 提取评论数据
        extracted_data = {
            "title": title,
        }

        for i, comment in enumerate(comments):
            entry = {
                "user_login": comment["user"]["login"],
                "created_at": comment["created_at"],
                "updated_at": comment["updated_at"],
                "body": comment["body"]
            }

            if i == 0:
                extracted_data.update(entry)  # 第一条评论直接加入
            else:
                extracted_data[f"answer_{i}"] = entry  # 后续评论作为回答

        # 添加到总提取数据中
        all_extracted_data.append(extracted_data)
    else:
        print(f"请求失败，状态码: {response.status_code}，对于标题: {title}")

# 保存提取的数据到新的 JSON 文件
with open('extracted_data.json', 'w') as f:
    json.dump(all_extracted_data, f, indent=4)

print("所有数据已提取并保存到 extracted_data.json")
