#pragma once

#include <cstddef>
#include <memory>

#include "arrow/api.h"

namespace graphar::ml {

// Sum of buffer sizes for ArrayData (recursive for children / dictionary).
inline size_t ArrayDataBytes(const std::shared_ptr<arrow::ArrayData>& data) {
  if (data == nullptr) return 0;
  size_t total = 0;
  for (const auto& buffer : data->buffers) {
    if (buffer != nullptr) total += static_cast<size_t>(buffer->size());
  }
  for (const auto& child : data->child_data) {
    total += ArrayDataBytes(child);
  }
  total += ArrayDataBytes(data->dictionary);
  return total;
}

inline size_t ScalarBytes(const std::shared_ptr<arrow::Scalar>& scalar) {
  if (scalar == nullptr || !scalar->is_valid) return 1;

  switch (scalar->type->id()) {
    case arrow::Type::BINARY:
    case arrow::Type::STRING:
    case arrow::Type::LARGE_BINARY:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::BINARY_VIEW:
    case arrow::Type::STRING_VIEW: {
      auto* binary = static_cast<const arrow::BaseBinaryScalar*>(scalar.get());
      return binary->value == nullptr ? 0
                                      : static_cast<size_t>(binary->value->size());
    }
    case arrow::Type::LIST:
    case arrow::Type::LARGE_LIST:
    case arrow::Type::LIST_VIEW:
    case arrow::Type::LARGE_LIST_VIEW:
    case arrow::Type::MAP:
    case arrow::Type::FIXED_SIZE_LIST: {
      auto* list = static_cast<const arrow::BaseListScalar*>(scalar.get());
      return list->value == nullptr ? 0 : ArrayDataBytes(list->value->data());
    }
    case arrow::Type::STRUCT: {
      auto* struct_scalar = static_cast<const arrow::StructScalar*>(scalar.get());
      size_t total = 0;
      for (const auto& child : struct_scalar->value) {
        total += ScalarBytes(child);
      }
      return total;
    }
    case arrow::Type::SPARSE_UNION:
    case arrow::Type::DENSE_UNION: {
      auto* union_scalar = static_cast<const arrow::UnionScalar*>(scalar.get());
      return ScalarBytes(union_scalar->child_value());
    }
    case arrow::Type::DICTIONARY: {
      auto* dict_scalar = static_cast<const arrow::DictionaryScalar*>(scalar.get());
      return ScalarBytes(dict_scalar->value.index) +
             ArrayDataBytes(dict_scalar->value.dictionary->data());
    }
    case arrow::Type::EXTENSION: {
      auto* ext_scalar = static_cast<const arrow::ExtensionScalar*>(scalar.get());
      return ScalarBytes(ext_scalar->value);
    }
    default: {
      auto* primitive =
          dynamic_cast<const arrow::internal::PrimitiveScalarBase*>(scalar.get());
      return primitive == nullptr ? sizeof(*scalar) : primitive->view().size();
    }
  }
}

}  // namespace graphar::ml
