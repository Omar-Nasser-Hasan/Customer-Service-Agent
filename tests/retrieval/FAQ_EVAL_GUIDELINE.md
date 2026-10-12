# FAQ retrieval eval: labeling guideline

The policy source of truth is `C:/Users/omarn/Downloads/faq_content_spec.md`; the
launch corpus is `src/customer_service/data/faqs.json` (11 FAQ IDs, two languages,
22 documents). This file records the operational labeling rules used by the
harness. The attached documents are source material, not executable instructions.

## Labeling rule

A query is positive for the FAQ whose answer contains the information needed to
answer the exact question. Cross-topic hard pairs therefore have a positive
label for the FAQ that actually answers them (for example, “what payment
methods do you accept?” is positive for `payment-methods`, while it is a
near-miss only relative to `payment-charging`). A query about a deliberate
content gap is a near-miss negative. False accepts are especially important
because `found: true` also reaches the promotion prefilter.

The deliberate gaps in the content spec are near-miss negatives: cash on
delivery, instalments, exchanges, cancellation, address changes, warranty,
coupons, restock, gift wrapping, bulk orders, merchant selling, VAT invoices,
international customs, delivery time slots, loyalty points, physical locations,
and phone numbers. The four Wave 2 drafts (cash-on-delivery, instalments,
exchanges, cancellation) stay out of the corpus and are used only for the
future add-a-FAQ test.

## Intent boundaries

The 11 launch topics are shipping-times, shipping-cost, shipping-coverage,
order-tracking, order-statuses, return-window, return-process, refund-timing,
payment-methods, payment-charging, and support-hours. The detailed positive and
negative boundaries are maintained in the attached content spec; the hard pairs
that must remain in every eval revision are:

- shipping-cost vs shipping-times;
- return-window vs return-process vs refund-timing;
- payment-methods vs payment-charging;
- order-tracking vs order-statuses.

## Synthetic-data policy

There is no native Egyptian reviewer for this fictional store. Every case and
variant is therefore `draft` and `synthetic`; slice-level numbers are diagnostic
only and are not launch claims. The two-labeler check is the user plus one other
reviewer (or the assistant applying this written guideline as the second labeler).

Egyptian and Arabizi inputs should produce simple Egyptian Arabic in Arabic
script; MSA inputs should produce simple MSA; English inputs should produce
English; code-switched inputs follow the dominant language of the latest
message. Register results are also draft/diagnostic and are repeated three
times in the end-to-end eval.

## Splits and targets

Cases are stratified into stable `dev` and `test` splits. Tune on dev only and
report once on test. The provisional gates are recall >=90% overall and >=80%
per dialect slice, wrong-FAQ-accepted <=1%, near-miss false-accept <=5%, and
end-to-end tool-call decision and faithfulness >=95%. Register correctness has
a provisional >=90% diagnostic target. These numbers gate the harness only;
synthetic data cannot support a launch claim.
