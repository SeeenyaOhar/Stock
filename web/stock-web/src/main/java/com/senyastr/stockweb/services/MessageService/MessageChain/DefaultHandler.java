package com.senyastr.stockweb.services.MessageService.MessageChain;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.repositories.ClassedMessageRepository;
import com.senyastr.stockweb.models.ClassedMessage;
import com.senyastr.stockweb.services.InquiryAnalyzer.InquirySummary;
import com.senyastr.stockweb.models.MessageClass;
import com.senyastr.stockweb.services.MessageService.MessageSelectorService.MessageSelectorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

/**
 * This handles the Inquiry in a default way: <br>just returns a message from the repository according to the message class.
 */
@Component("messageChainDefaultHandler")
public class DefaultHandler extends BaseHandler{
    MessageSelectorService messageSelectorService;
    ClassedMessageRepository messageRepository;
    @Autowired
    public DefaultHandler(MessageSelectorService service, ClassedMessageRepository messageRepository){
        this.messageSelectorService = service;
        this.messageRepository = messageRepository;
    }
    @Override
    public Message handle(InquirySummary summary){
        int _class = summary.getMessageClass();
        MessageClass adaptedClass = switch (_class) {
            case 6 -> MessageClass.FEEDBACK_NEUTRAL;
            case 7 -> MessageClass.CHECKOUT;
            case 8 ->MessageClass.CHECKOUT_REQUEST;
            case 9 -> MessageClass.RECOMMENDATION;
            default -> MessageClass.values()[_class];
        };
        Iterable<ClassedMessage> messages = messageRepository.findClassedMessageByMessageClass(adaptedClass);
        ArrayList<ClassedMessage> messages_ar = new ArrayList<ClassedMessage>();
        for (var v : messages){
            messages_ar.add(v);
        }
        return messageSelectorService.select(messages_ar);
    }
}
