# Ethics and data protection

## Data sources and collection

Only publicly available information was collected: channel metadata, video titles and descriptions, top-level public comments, and the public SEBI registers. Collection used the official YouTube Data API v3 under its terms of service and, for the registers, one polite request about every 1.5 seconds with a user-agent that identifies the research purpose and a contact address.

## Personal data

- Commenter identifiers were replaced by salted hashes at collection time; commenter names were never stored.
- Personal contact details in the SEBI listings (e-mail, telephone, address, contact person) were discarded by the scraper and are not used.
- The released tables contain hashed account identifiers, categorical labels and numeric features only. They contain no names, URLs or free text. The hashing salt is held privately; without it the identifiers cannot be linked to channels.

## Interpretation and use

- Unregistered status is not evidence of unlawful activity. Educators, news channels and brokers' education arms need no adviser registration.
- The classifiers here estimate resemblance to registered advisers. The accompanying manuscript reports that this ranking did not identify adviser-like behaviour in blind human validation, and that a pilot model trained on the coded behaviour has wide uncertainty. Neither should be used to accuse or sanction any account. Any regulatory use would require human review and due process.
- Results are reported in aggregate. No individual channel is named in the manuscript or in the released data.

## Enforcement orders

The scoping scripts read public SEBI orders only to count how many name identifiable YouTube channels. Many orders are interim and may be appealed, so no order-based labels are used or released.

## Review status

Institutional ethics review status: [to be completed before submission].
