class Observable:
    def __init__(self):
        self.observers = []

    def register_observer(self, observer):
        if observer in self.observers:
            raise ValueError("Observer already registered")
        self.observers.append(observer)

    def unregister_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self, data):
        for observer in self.observers:
            observer.update(data)


class Observer:
    def update(self, data):
        raise NotImplementedError("Subclasses must implement update()")
