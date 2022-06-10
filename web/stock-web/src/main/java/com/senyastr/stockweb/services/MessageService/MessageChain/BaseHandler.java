package com.senyastr.stockweb.services.MessageService.MessageChain;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.services.InquiryAnalyzer.InquirySummary;

public class BaseHandler implements Handler{
    protected Handler next;
    @Override
    public void setNext(Handler handler) {
        this.next = handler;
    }
    @Override
    public Message handle(InquirySummary summary) {
        if (next != null){
            return next.handle(summary);
        }
        else{
            return null;
        }
    }
}
