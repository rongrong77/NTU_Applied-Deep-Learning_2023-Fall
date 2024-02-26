import matplotlib.pyplot as plt
import json


with open('E:/ADL23-HW2_R/best/eval_result.json', 'r', encoding= "utf-8") as file:
    data = json.load(file)

epochs = list(data.keys())
rouge1_f1_scores = [data[str(epoch)]['rouge-1']['f'] for epoch in epochs]
rouge2_f1_scores = [data[str(epoch)]['rouge-2']['f'] for epoch in epochs]
rougel_f1_scores = [data[str(epoch)]['rouge-l']['f'] for epoch in epochs]

plt.figure(figsize=(10, 6))

plt.plot(epochs, rouge1_f1_scores, label='Rouge-1 F1 Score', marker='o')
plt.plot(epochs, rouge2_f1_scores, label='Rouge-2 F1 Score', marker='o')
plt.plot(epochs, rougel_f1_scores, label='Rouge-L F1 Score', marker='o')

plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score for Rouge Metrics over Epochs')
plt.legend()

plt.grid()
plt.show()