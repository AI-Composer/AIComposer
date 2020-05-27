# structure
shared.eventss_rnn_graph.py
分了三个模式：train,eval,generate.

encoder_decoder?



执行train以后：
events_rnn_graph.get_build_graph_fn()来建立一个tf的图结构。 config？
然后通过events_rnn_train.run_train()来跑训练。


events_rnn_train.run_train():
有loss,perplexity,accuracy,train_op这么几个训练参数。