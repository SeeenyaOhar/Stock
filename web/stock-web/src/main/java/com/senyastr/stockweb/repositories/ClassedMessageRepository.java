package com.senyastr.stockweb.repositories;

import com.senyastr.stockweb.models.ClassedMessage;
import com.senyastr.stockweb.models.MessageClass;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ClassedMessageRepository extends CrudRepository<ClassedMessage, Integer> {
    Iterable<ClassedMessage> findClassedMessageByMessageClass(MessageClass messageClass);
}