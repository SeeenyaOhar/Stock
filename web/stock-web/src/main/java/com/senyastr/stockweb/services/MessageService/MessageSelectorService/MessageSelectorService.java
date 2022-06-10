package com.senyastr.stockweb.services.MessageService.MessageSelectorService;

import com.senyastr.stockweb.models.Message;
import com.senyastr.stockweb.models.ClassedMessage;

import java.util.List;

public interface MessageSelectorService {
    Message select(List<ClassedMessage> messageList);
}
