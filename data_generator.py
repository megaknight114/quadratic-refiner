import torch
def generate_quadratic_data(num_samples=100):
    # 随机生成根 x1, x2 在 [-1, 1] 之间
    x1 = torch.rand(num_samples) * 2 - 1  
    x2 = torch.rand(num_samples) * 2 - 1  
    a = torch.rand(num_samples) * 2 - 1
    a = torch.where(a == 0, torch.ones_like(a), a) 
    b = -a * (x1 + x2)  
    c = a * x1 * x2    
    X = torch.stack([a, b, c], dim=1)
    y = torch.max(x1, x2).unsqueeze(1)  
    return X, y

def save_data(X, y, filepath='/home/xuzonghuan/quadratic-refiner/quadratic_data.pt'):
    torch.save((X, y), filepath)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    X, y = generate_quadratic_data(num_samples=100)  
    save_data(X, y)