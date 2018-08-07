TrainImage.exe -xml=train.xml


train.xml
<train>
  <positive>path to the positive images</positive>
  <negative>path to the negative images</negative>
  <number>500</number>
  <complexity>1000</complexity>
  <bow_output>bow.bin</bow_output>
  <machine_output>machine.bin</machine_output>
</train>