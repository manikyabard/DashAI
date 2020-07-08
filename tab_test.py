from tabular.learner import DashTabularLearner

print('So far, so good...')
learn = DashTabularLearner.create_tabular_learner()
print('Yay!')

learn.fit_one_cycle(2)
print(learn)