package com.senyastr.stockweb.services.MessageService;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class MessageServiceTest {
    MessageService service;
    @Autowired
    MessageServiceTest(MessageService messageService){
        this.service = messageService;
    }
    @Test
    void messageServiceTest1(){
        var message= "Hi, your product was amazing! I really like it!";
        System.out.println("Message: " + message + '\n');
        System.out.println("Answer: " + service.process(message).getMessage());
    }
    @Test
    void messageServiceTest2(@Autowired MessageService service){
        var message = "Hi!";
        System.out.println("Message: " + message + '\n');
        System.out.println("Answer: " + service.process(message).getMessage());
        message = "I would like to buy an IPhone 12!";
        System.out.println("Message: " + message + '\n');
        System.out.println("Answer: " + service.process(message).getMessage());
        message = "How much does it cost?";
        System.out.println("Message: " + message + '\n');
        System.out.println("Answer: " + service.process(message).getMessage());
        message = "Do you deliver to Ukraine?";
        System.out.println("Message: " + message + '\n');
        System.out.println("Answer: " + service.process(message).getMessage());
    }
}
