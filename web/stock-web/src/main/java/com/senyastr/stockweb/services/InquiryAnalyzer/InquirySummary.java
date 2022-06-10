package com.senyastr.stockweb.services.InquiryAnalyzer;

import lombok.Data;

@Data
public class InquirySummary {
    private int messageClass;
    public InquirySummary(int _class){
        this.messageClass = _class;
    }
}
