import pydot
from torch.autograd import Variable


# From https://gist.github.com/apaszke/01aae7a0494c55af6242f06fad1f8b70
class Graph(object):
    def __init__(self):
        self.seen = set()
        self.dot = pydot.Dot("resnet", graph_type="digraph", rankdir="TB")
        self.style_params = {"shape": "octagon",
                             "fillcolor": "gray",
                             "style": "filled",
                             "label": "",
                             "color": "none"}
        self.style_layers = {"shape": "box",
                             "fillcolor": "blue",
                             "style": "filled",
                             "label": "",
                             "color": "none"}

    def addnodes(self, var):
        if var not in self.seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                self.style_params["label"] = value
                self.dot.add_node(pydot.Node(str(id(var)), **self.style_params))

            else:
                value = str(type(var).__name__)
                self.style_layers["label"] = value
                if value == "ConvNd":
                    self.style_layers["fillcolor"] = "cyan"
                elif value == "BatchNorm":
                    self.style_layers["fillcolor"] = "darkseagreen"
                elif value == "Threshold":
                    self.style_layers["fillcolor"] = "crimson"
                    self.style_layers["label"] = "ReLU"
                elif value == "Add":
                    self.style_layers["fillcolor"] = "darkorchid"
                elif value == "AvgPool2d":
                    self.style_layers["fillcolor"] = "gold"
                elif value == "Linear":
                    self.style_layers["fillcolor"] = "chartreuse"
                elif value == "View":
                    self.style_layers["fillcolor"] = "brown"
                else:
                    self.style_layers["fillcolor"] = "aquamarine"

                self.dot.add_node(pydot.Node(str(id(var)), **self.style_layers))

            self.seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    # if not isinstance(u[0], Variable):
                    self.dot.add_edge(pydot.Edge(str(id(u[0])), str(id(var))))
                    self.addnodes(u[0])

    def draw(self, var):
        self.addnodes(var.creator)

    def save(self, file_name="network.svg"):
        ext = file_name[file_name.rfind(".")+1:]
        with open(file_name, 'wb') as fid:
            fid.write(self.dot.create(format=ext))
        # self.dot.save(filename=file_name)
        print ("Save Network Graph Done!")
