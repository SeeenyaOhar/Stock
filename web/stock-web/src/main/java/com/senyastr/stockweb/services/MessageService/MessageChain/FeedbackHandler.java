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

@Component
public class FeedbackHandler extends BaseHandler{
    private final int FEEDBACK_MESSAGE_CLASS = 6;
    private boolean isFeedback(InquirySummary summary){
        return summary.getMessageClass() == FEEDBACK_MESSAGE_CLASS;
    }
    private final MessageSelectorService selectorService;
    private final ClassedMessageRepository messageRepository;
    @Autowired
    public FeedbackHandler(MessageSelectorService selectorService, ClassedMessageRepository messageRepository){
        this.selectorService = selectorService;
        this.messageRepository = messageRepository;
    }
    @Override
    public Message handle(InquirySummary summary){
        if (isFeedback(summary)){
            Iterable<ClassedMessage> messages =  messageRepository.findClassedMessageByMessageClass(MessageClass.FEEDBACK_NEUTRAL);
            ArrayList<ClassedMessage> messages_ar = new ArrayList<ClassedMessage>();
            for (var v : messages){
                messages_ar.add(v);
            }
            return selectorService.select(messages_ar);
        }
        else{
            return super.handle(summary);
        }
    }
}
