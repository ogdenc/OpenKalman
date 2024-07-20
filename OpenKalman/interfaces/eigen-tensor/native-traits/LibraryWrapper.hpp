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
 * \brief Traits for \ref Eigen3::EigenTensorWrapper (an alias for LibraryWrapper in the Eigen library)
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_LIBRARYWRAPPER_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_LIBRARYWRAPPER_HPP


namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_tensor_general T, OpenKalman::Eigen3::eigen_tensor_general L>
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>>
#else
  template<typename T, typename L>
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_tensor_general<T> and OpenKalman::Eigen3::eigen_tensor_general<L>>>
#endif
    : Eigen::internal::traits<std::decay_t<T>> {};


#ifdef __cpp_concepts
  template<typename T, OpenKalman::Eigen3::eigen_tensor_general L> requires (not OpenKalman::Eigen3::eigen_tensor_general<T>)
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>>
#else
  template<typename T, typename L>
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>, std::enable_if_t<
    not OpenKalman::Eigen3::eigen_tensor_general<T> and OpenKalman::Eigen3::eigen_tensor_general<L>>>
#endif
  {
    using Scalar = scalar_type_of_t<T>;
    using StorageKind = Eigen::Dense;
    using Index = Eigen::Index;
    static const int NumDimensions = index_count_v<T>;
    static const int Layout = OpenKalman::layout_of_v<T> == OpenKalman::Layout::right ? Eigen::RowMajor : Eigen::ColMajor;

  private:

    using Ix_array = std::array<Index, NumDimensions>;
    using ElementRef = decltype(OpenKalman::get_component(std::declval<std::add_lvalue_reference_t<T>>(), std::declval<Ix_array>()));
    static constexpr auto lvalue_bit = std::is_same_v<ElementRef, std::decay_t<ElementRef>&> ? Eigen::LvalueBit : 0x0;

  public:

    enum {
      Options = Layout,
      Flags = Eigen::internal::compute_tensor_flags<Scalar, Options>::ret | lvalue_bit,
    };

    using PointerType = Scalar*;
  };

} // OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_LIBRARYWRAPPER_HPP