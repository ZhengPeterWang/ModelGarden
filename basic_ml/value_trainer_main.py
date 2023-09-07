from value_trainer import ValueTrainer
from value_backprop import MLP

def main():
  model = MLP(3, [4, 4, 1])
  xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
  ys = [1.0, -1.0, -1.0, 1.0]

  trainer = ValueTrainer()
  trainer.train(model, xs, ys)

  y_pred = [model(x) for x in xs]
  print(y_pred)

if __name__ == '__main__':
  main()
