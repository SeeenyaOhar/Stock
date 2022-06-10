package com.senyastr.stockweb.services.MessageService.MessageSelectorService;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.models.ClassedMessage;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Random;
@Service
public class RandomSelectorService implements MessageSelectorService{
    @Override
    public Message select(List<ClassedMessage> messageList) {
        Random random = new Random();
        int bound = messageList.size();
        int index = random.nextInt(0, bound);
        return messageList.get(index);
    }
}
