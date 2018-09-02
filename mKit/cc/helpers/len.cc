template<typename T>
int len(T data) {
  int length = sizeof(data)/sizeof(*data);
  return length;
}
