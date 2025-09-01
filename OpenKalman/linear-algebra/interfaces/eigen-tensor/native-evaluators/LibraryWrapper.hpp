/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen evaluator of /ref LibraryWrapper for the Eigen tensor library
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_EVALUATORS_LIBRARYWRAPPER_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_EVALUATORS_LIBRARYWRAPPER_HPP


namespace Eigen
{
  // ------------------------ //
  //  Eigen::TensorEvaluator  //
  // ------------------------ //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename NestedObject, typename Device>
#else
    template<typename NestedObject, typename Device, typename = void>
#endif
    struct EigenTensorWrapperEvaluator
    {
      using Index = Eigen::Index;

    private:

      template<typename Arg, std::size_t...is>
      static auto make_dim(const Arg& arg, std::index_sequence<is...>) { return Eigen::Sizes<OpenKalman::index_dimension_of_v<Arg, is>...> {}; };

#ifdef __cpp_concepts
      template<OpenKalman::has_dynamic_dimensions Arg, std::size_t...is>
#else
      template<typename Arg, std::size_t...is, std::enable_if_t<OpenKalman::has_dynamic_dimensions<Arg>, int> = 0>
#endif
      static auto make_dim(const Arg& arg, std::index_sequence<is...>)
      {
        return Eigen::DSizes<Index, OpenKalman::index_count_v<Arg>> {OpenKalman::get_index_dimension_of<is>(arg)...};
      };

    public:

      using Scalar = OpenKalman::scalar_type_of_t<NestedObject>;
      using CoeffReturnType = Scalar;
      using PacketReturnType = typename PacketType<CoeffReturnType, Device>::type;
      using Dimensions = decltype(make_dim(std::declval<NestedObject>(), std::make_index_sequence<OpenKalman::index_count_v<NestedObject>>{}));
      static const int PacketSize =  PacketType<CoeffReturnType, Device>::size;
      using TensorPointerType = Scalar*;
      using Storage = StorageMemory<Scalar, Device>;
      using EvaluatorPointerType = typename Storage::Type;
      using TensorBlock = internal::TensorBlockNotImplemented;

      // NumDimensions is -1 for variable dim tensors

      static const int NumCoords {OpenKalman::index_count_v<NestedObject> == OpenKalman::dynamic_size ?
        0 : static_cast<int>(OpenKalman::index_count_v<NestedObject>)};

      enum {
        IsAligned          = false,
        PacketAccess       = false,
        BlockAccess        = false,
        PreferBlockAccess  = false,
        data_layout             = OpenKalman::layout_of_v<NestedObject> == OpenKalman::data_layout::right ? Eigen::RowMajor : Eigen::ColMajor,
        CoordAccess        = OpenKalman::index_count_v<NestedObject> > 0 and OpenKalman::index_count_v<NestedObject> != OpenKalman::dynamic_size,
        RawAccess          = false,
      };

      EigenTensorWrapperEvaluator(const NestedObject& arg, const Device& device)
        : m_dims {make_dim(arg, std::make_index_sequence<OpenKalman::index_count_v<NestedObject>>{})} {}


      constexpr auto dimensions() const { return m_dims; }

      constexpr bool evalSubExprsIfNeeded(EvaluatorPointerType dest) { return false; }

#ifdef EIGEN_USE_THREADS
      template <typename EvalSubExprsCallback>
      constexpr void evalSubExprsIfNeededAsync(EvaluatorPointerType dest, EvalSubExprsCallback done) {}
#endif  // EIGEN_USE_THREADS

      EIGEN_STRONG_INLINE void cleanup() {}

    private:

      Dimensions m_dims;

    };


#ifdef __cpp_concepts
    template<OpenKalman::directly_accessible NestedObject, typename Device> requires
      (not OpenKalman::Eigen3::eigen_tensor_general<NestedObject>)
    struct EigenTensorWrapperEvaluator<NestedObject, Device>
#else
    template<typename NestedObject, typename Device>
    struct EigenTensorWrapperEvaluator<NestedObject, Device, std::enable_if_t<
      OpenKalman::directly_accessible<NestedObject> and not OpenKalman::Eigen3::eigen_tensor_general<NestedObject>>>
#endif
      : TensorEvaluator<std::decay_t<decltype(OpenKalman::to_native_matrix<Tensor<double,0>>(std::declval<NestedObject>()))>, Device>
    {
      EigenTensorWrapperEvaluator(const NestedObject& arg, const Device& device)
        : TensorEvaluator<std::decay_t<decltype(OpenKalman::to_native_matrix<Tensor<double,0>>(std::declval<NestedObject>()))>, Device>
            {OpenKalman::to_native_matrix<Tensor<double,0>>(arg), device} {}
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::eigen_tensor_general NestedObject, typename Device>
    struct EigenTensorWrapperEvaluator<NestedObject, Device>
#else
    template<typename NestedObject, typename Device>
    struct EigenTensorWrapperEvaluator<NestedObject, Device, std::enable_if_t<OpenKalman::Eigen3::eigen_tensor_general<NestedObject>>>
#endif
      : TensorEvaluator<NestedObject, Device>
    {
      using Base = TensorEvaluator<NestedObject, Device>;
      EigenTensorWrapperEvaluator(const NestedObject& t, const Device& device) : Base {t, device} {}
    };

  }


  template<typename NestedObject, typename LibraryObject, typename Device>
  struct TensorEvaluator<OpenKalman::internal::LibraryWrapper<NestedObject, LibraryObject>, Device>
    : detail::EigenTensorWrapperEvaluator<std::remove_reference_t<NestedObject>, Device>
  {
    using XprType = OpenKalman::internal::LibraryWrapper<NestedObject, LibraryObject>;
    using Base = detail::EigenTensorWrapperEvaluator<std::remove_reference_t<NestedObject>, Device>;
    TensorEvaluator(const XprType& t, const Device& device) : Base {t.nested_object(), device} {}
  };


  template<typename NestedObject, typename LibraryObject, typename Device>
  struct TensorEvaluator<const OpenKalman::internal::LibraryWrapper<NestedObject, LibraryObject>, Device>
    : detail::EigenTensorWrapperEvaluator<const std::remove_reference_t<NestedObject>, Device>
  {
    using XprType = OpenKalman::internal::LibraryWrapper<const NestedObject, LibraryObject>;
    using Base = detail::EigenTensorWrapperEvaluator<const std::remove_reference_t<NestedObject>, Device>;
    TensorEvaluator(const XprType& t, const Device& device) : Base {t.nested_object(), device} {}
  };

} // Eigen::internal

#endif