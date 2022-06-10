package com.senyastr.stockweb.services.MessageService;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.repositories.ClassedMessageRepository;
import com.senyastr.stockweb.services.InquiryAnalyzer.InquiryAnalyzerService;
import com.senyastr.stockweb.services.InquiryAnalyzer.InquirySummary;
import com.senyastr.stockweb.services.MessageService.MessageChain.Handler;
import com.senyastr.stockweb.services.MessageService.MessageSelectorService.MessageSelectorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
public class MessageServiceBERT implements MessageService{
    InquiryAnalyzerService analyzerService;
    ClassedMessageRepository messageRepository;
    MessageSelectorService selector;
    ApplicationContext context;
    @Autowired
    public MessageServiceBERT(InquiryAnalyzerService analyzerService, ClassedMessageRepository messageRepository,
                              MessageSelectorService selector, ApplicationContext context){
        this.analyzerService = analyzerService;
        this.messageRepository = messageRepository;
        this.selector = selector;
        this.context = context;
    }

    @Override
    public Message process(String message) {
        InquirySummary summary = analyzerService.analyzeInquiry(message);
        // based on the class handlers return a message
        Map<String, Handler> handlersMapping = context.getBeansOfType(Handler.class);
        String defaultHandlerBeanName = "messageChainDefaultHandler";
        // default handler is always in the end
        Handler defaultHandler = (Handler) context.getBean(defaultHandlerBeanName);
        List<Handler> handlers = new ArrayList<Handler>();
        handlersMapping.forEach((s, handler) -> {
            if (!Objects.equals(s, defaultHandlerBeanName)){
                handlers.add(handler);
            }
        });
        Handler initial = handlers.get(0);
        Handler previous = initial;
        for (int i = 1; i < handlers.size(); i++){
            Handler next = handlers.get(i);
            previous.setNext(next);
            previous = next;
        }
        previous.setNext(defaultHandler);
        return initial.handle(summary);
    }
}
