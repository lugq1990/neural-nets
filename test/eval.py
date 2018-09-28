# -*- coding:utf-8 -*-

def e(model, data, label):
    model.fit(data, label)
    acc = model.evaluate(data, label)
    pred = model.predict(data)
    print('Acc:', acc)
    print('prediction: ', pred[:10])
    model.plot_acc()