class Log:
    def __init__(self):
        self.trace = []
        self.events = {'deal': [], 'skip': [], 'cheat': [], 'reject': [], 'comm': []}
        self.user_message_log = []
        self.attacks = []

    def update_trace(self, event):
        # self.trace.append(event)
        pass

    def update_events(self, time, event):
        # self.events[event[0]].append((time, event))
        pass

    def update_attacks(self, attack):
        self.attacks.append(attack)
        # pass

    # def update_user_message_log(self, users):
    #     new_messages = [{'deal': set(), 'cheat': set()} for _ in users]
    #     for label, u in enumerate(users):
    #         new_messages[label]['deal'].update(u.messages['deal'])
    #         new_messages[label]['cheat'].update(u.messages['cheat'])
    #     self.user_message_log.append(new_messages)