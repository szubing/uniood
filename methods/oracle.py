from methods.source_only import SourceOnly

class Oracle(SourceOnly):
    require_source = False
    require_target = True

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def forward(self, batched_inputs):
        target_images = batched_inputs['target_images'].to(self.device)
        target_labels = batched_inputs['target_labels'].to(self.device)
        
        mask = target_labels < self.num_classes
        features = self.backbone(target_images[mask])
        logit = self.classifier(features)

        loss = self.criterion(logit, target_labels[mask])

        return loss

        
