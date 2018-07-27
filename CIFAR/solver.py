import torch

class Solver(object):
    """Handle standard, boilerplate training and evaluation loops for PyTorch 
    models.

    Parameters
    ----------
    model : torch.nn.Module
        Learning model we wish to train.
    optimizer : torch.nn.Optimizer
        Stochastic optimizer acting on the parameters of model.
    loss_fn : function
        Function which outputs a loss, given model outputs and a set of labels, 
        as a Tensor (i.e., supporting autograd backwards passes).
    initializer : function
        Function which takes a model parameter and initializes its weights. Defaults
        to default_initializer (implemented below).
    device : torch.device
        Device used to run the training/evaluation (i.e., where we will move data).
        Defaults to 'cuda:0' if CUDA is enabled, 'cpu' otherwise.
        NOTE: The user must ensure that the model is already located on the 
        correct device (since this must be done prior to optimizer definition),
        this class will not move the model onto the GPU!
    verbose : bool
        Whether or not to print training loss (see print_every) and accuracies 
        every epoch. Default to False.
    print_every : int
        How often to print the current loss, in terms of mini-batch iterations. 
        Defaults to 100
    """
    def __init__(self, model, optimizer, loss_fn, initializer=None,
                 device=None, verbose=False, print_every=100):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if initializer is None:
            initializer = default_initializer
        self.initializer = initializer
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.verbose = verbose
        self.print_every = print_every
        self.num_epochs = 0
        self.num_iter = 0
        self.loss_history = []
        self.train_acc_history = []
        self.eval_acc_history = []

    def reset_model(self):
        self.model.apply(self.initializer)
        self.num_epochs = 0
        self.num_iter = 0
        self.loss_history = []
        self.train_acc_history = []
        self.eval_acc_history = []

    def train(self, data_loader, num_epochs):
        """
        Parameters
        ----------
        data_loader : torch.DataLoader
            Iterates over tuples of the form (inputs, labels), where these are the
            mini-batches of data and labels, respectively.
        num_epochs : int
            Number of training epochs.
        """
        for epoch in range(num_epochs):
            self._train_one_epoch(data_loader)
            if self.verbose:
                self._print_accuracy()

    def evaluate(self, data_loader):
        """
        Parameters
        ----------
        data_loader : torch.DataLoader
            Iterates over tuples of the form (inputs, labels), where these are the
            mini-batches of data and labels, respectively.
        """
        self.model.eval()
        num_correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)

                outputs = self.model(inputs)
                _, predictions = torch.max(outputs.data, 1)
                total += labels.size(0)
                num_correct += (predictions == labels).sum().item()
        return num_correct/total

    def train_eval_loop(self, train_data_loader, eval_data_loader, num_epochs):
        for epoch in range(num_epochs):
            self._train_one_epoch(train_data_loader)
            eval_accuracy = self.evaluate(eval_data_loader)
            self.eval_acc_history.append(eval_accuracy)
            if self.verbose:
                self._print_accuracy()

    def _train_one_epoch(self, data_loader):
        self.model.train()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.long().to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            _, predictions = torch.max(outputs.data, 1)
            num_correct = (predictions == labels).sum().item()
            self.train_acc_history.append(num_correct/labels.size(0))

            if self.verbose and self.num_iter % self.print_every == 0:
                print('(Iteration %d) loss: %f' % (self.num_iter, self.loss_history[-1]))
            self.num_iter += 1
        self.num_epochs += 1

    def _print_accuracy(self):
        if len(self.train_acc_history) == 0:
            return
        print_string = '(Epoch %d) train acc: %f' % (self.num_epochs, self.train_acc_history[-1])
        if len(self.eval_acc_history) > 0:
            print_string += '; val acc: %f' %(self.eval_acc_history[-1])
        print(print_string)




def default_initializer(param):
    """Initialize the weights/biases of a model parameter. Convolutional and 
    fully-connected layers will be initialized using the standard Glorot normal
    scheme for weights. Batch-norm will be initialized to a uniform [0, 1] 
    distribution. All biases will be initialized to 0.
    """
    class_name = param.__class__.__name__
    if 'Conv' in class_name:
        torch.nn.init.xavier_normal_(param.weight)
    elif 'Linear' in class_name:
        torch.nn.init.xavier_normal_(param.weight)
    elif 'BatchNorm' in class_name:
        torch.nn.init.uniform_(param.weight)
    if hasattr(param, 'bias') and param.bias is not None:
        torch.nn.init.constant_(param.bias, 0)