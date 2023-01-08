package com.senyastr.stockweb.services.InquiryAnalyzer;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class InquiryAnalyzerTest {
    InquiryAnalyzerService service;
    @Autowired
    InquiryAnalyzerTest(InquiryAnalyzerService analyzerService){
        this.service = analyzerService;
    }
    @Test
    void inquiryAnalyzerTest1(){
        var message = "Could you recommend a phone please?";
        System.out.println("Testing this message and returning the class for this message:\n" + message);
        System.out.println(service.analyzeInquiry(message));
    }
}
