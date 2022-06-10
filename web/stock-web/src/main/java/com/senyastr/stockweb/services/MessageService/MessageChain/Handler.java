package com.senyastr.stockweb.services.MessageService.MessageChain;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.services.InquiryAnalyzer.InquirySummary;

public interface Handler {
    void setNext(Handler handler);
    Message handle(InquirySummary summary);
}
