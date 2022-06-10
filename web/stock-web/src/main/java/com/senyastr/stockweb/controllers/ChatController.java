package com.senyastr.stockweb.controllers;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.models.Sender;
import com.senyastr.stockweb.services.MessageService.MessageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@Controller
@RequestMapping("/chat")
@SessionAttributes("messages")
public class ChatController {
    final private MessageService messageService;
    final private String MESSAGES_ATTRIBUTE = "messages";
    final private String NUMBER_MESSAGES_ATTRIBUTE = "messageNumber";
    @Autowired
    public ChatController(MessageService messageService){
        this.messageService = messageService;
    }
    @ModelAttribute(MESSAGES_ATTRIBUTE)
            public ArrayList<Message> getMessages()
    {
        return new ArrayList<Message>();
    }
    @GetMapping
    public String getChat(Model model){
        int numberOfMessages = 0;
        List<Message> messages = (List<Message>)model.getAttribute(MESSAGES_ATTRIBUTE);
        if (messages == null){
            var arrayList = new ArrayList<Message>();
            model.addAttribute(MESSAGES_ATTRIBUTE, arrayList);
        }
        else{
            numberOfMessages = messages.size();
        }

        model.addAttribute(NUMBER_MESSAGES_ATTRIBUTE, numberOfMessages);
        model.addAttribute("newMessage", new Message());
        return "chat";
    }
    /**
     * Receives a message. The message is processed by <b>the Stock Inquiry Analyzer</b>.<br>
     * Once it's analyzed it returns:<br>
     * 1. JSON mapping.<br>
     * 2. Contains two key-value pairs: String message, int classification(0..9)<br>
     *
     * <i>E.g.<br>
     * {<br>
     *     "message": "Hey, do you deliver sneakers to Ukraine?",<br>
     *     "classification": 3 <br>
     * }<br>
     * </i>
     * Here number 3 means classification number 3 corresponding to <b>DELIVERY</b> class.
     * The message for such classification is fetched from the database and is shown to the user.
     */
    @PostMapping
    public String messageReceived(@ModelAttribute("newMessage") Message message, @ModelAttribute(MESSAGES_ATTRIBUTE) ArrayList<Message> messages,
                                  Model model){
        message.setSender(Sender.CLIENT);
        messages.add(message);
        Message botMessage = messageService.process(message.getMessage());
        messages.add(botMessage);
        model.addAttribute(MESSAGES_ATTRIBUTE, messages);
        return "chat";
    }
}
