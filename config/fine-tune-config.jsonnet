freeze
{"trainer":
  {
  	"no_grad":
	[".*text_field_embedder.*", ".*encoder.*"],
    [".*text_field_embedder.*", ".*layer.*", ".*matrix.*"]
    [".*text_field_embedder.*", ".*_embedding_proj.*", ".*_highway.*", ".* _phrase_layer.*", ".*_encoding_proj.*"]
  }
}

trainer:  should_log_learning_rate

allennlp find-lr ./config.jsonnet -s ./find_lr --start-lr 0.0000001 --end-lr 0.5


class _Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_2 = torch.nn.Linear(10, 5)

    def forward(self, inputs):  # pylint: disable=arguments-differ
        pass

class _Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 10)
        self.linear_3 = torch.nn.Linear(10, 5)

    def forward(self, inputs):  # pylint: disable=arguments-differ
        pass
        
得到net2的模型，用来初始化net1


pretrain
{"model":
	
"initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*linear_layers.*bias", {"type": "constant", "val": 0}],
      [".*weight_ih.*", {"type": "xavier_normal"}],
      [".*weight_hh.*", {"type": "orthogonal"}],
      [".*bias.*", {"type": "constant", "val": 0}],
      [".*matcher.*match_weights.*", {"type": "kaiming_normal"}],
      [
      	  "linear_1.weight|linear_2.weight", # linear_1和linear_2使用预训练模型参数  # 正则表达式 linear_1.*  # linear_1.weight|linear_1.bias
		      {
		          "type": "pretrained",
		          "weights_file_path": "temp_file",
		          "parameter_name_overrides": "linear_2.weight": "linear_3.weight" # 其中linear_2用linear_3的进行初始化，但lay大小要一致
		        }
      ]
    ]
}


allennlp find-lr ./qanet.jsonnet -s ./find_lr --start-lr 0.00000001 --end-lr 0.5 --num-batches 10 -f
{"model":
"initializer": [
      	[
      	  ".*weight|.*bias",
		      {
		          "type": "pretrained",
		          "weights_file_path": "./best.th"
		        }
        ]
    ]
}

no_char:
"initializer": [
      	[
[".*_phrase_layer.*|.*_matrix.*|.*_modeling.*|.*_predictor.*"] 
 {
		          "type": "pretrained",
		          "weights_file_path": "./best.th"
		        }
        ]
    ]