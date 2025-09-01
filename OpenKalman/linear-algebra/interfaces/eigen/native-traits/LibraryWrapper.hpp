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
 * \brief Traits for \ref Eigen3::EigenWrapper (an alias for LibraryWrapper in the Eigen library)
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_LIBRARYWRAPPER_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_LIBRARYWRAPPER_HPP


namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general T, OpenKalman::Eigen3::eigen_general L>
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>>
#else
  template<typename T, typename L>
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<T> and OpenKalman::Eigen3::eigen_general<L>>>
#endif
    : Eigen::internal::traits<std::decay_t<T>>
  {
  private:

    using NestedTraits = Eigen::internal::traits<std::decay_t<T>>;
    using GetRes = decltype(OpenKalman::get_component(std::declval<std::add_lvalue_reference_t<T>>(),
      std::declval<std::size_t>(), std::declval<std::size_t>()));

    static constexpr auto lvalue_bit = (NestedTraits::Flags & Eigen::LvalueBit) != 0x0 and (OpenKalman::Eigen3::eigen_dense_general<T, true> or
      (std::is_lvalue_reference_v<GetRes> and not std::is_const_v<std::remove_reference_t<GetRes>>)) ? Eigen::LvalueBit : 0x0;

  public:

    using XprKind = std::conditional_t<OpenKalman::Eigen3::eigen_array_general<T, true>, Eigen::ArrayXpr, Eigen::MatrixXpr>;
    using Scalar = OpenKalman::scalar_type_of_t<T>;
    using StorageKind = Eigen::Dense;
    using StorageIndex = Eigen::Index;
    enum {
      Flags = (NestedTraits::Flags & ~Eigen::NestByRefBit & ~Eigen::LvalueBit) | lvalue_bit,
    };
  };


#ifdef __cpp_concepts
  template<typename T, OpenKalman::Eigen3::eigen_general L> requires (not OpenKalman::Eigen3::eigen_general<T>)
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>>
#else
  template<typename T, typename L>
  struct native_traits<OpenKalman::internal::LibraryWrapper<T, L>, std::enable_if_t<
    not OpenKalman::Eigen3::eigen_general<T> and OpenKalman::Eigen3::eigen_general<L>>>
#endif
  {
  private:

    using ElementRef = decltype(OpenKalman::get_component(std::declval<std::add_lvalue_reference_t<T>>(), 0, 0));
    static constexpr auto lvalue_bit = std::is_same_v<ElementRef, std::decay_t<ElementRef>&> ? Eigen::LvalueBit : 0x0;
    static constexpr auto layout_bit = OpenKalman::layout_of_v<T> == OpenKalman::data_layout::right ? Eigen::RowMajorBit : 0x0;
    static constexpr auto direct = OpenKalman::directly_accessible<T> ? Eigen::DirectAccessBit : 0x0;

#ifdef __cpp_concepts
    template<typename Arg>
#else
    template<typename Arg, typename = void>
#endif
    struct Strides { using type = std::decay_t<decltype(OpenKalman::internal::strides(std::declval<Arg>()))>; };

#ifdef __cpp_concepts
    template<typename Arg> requires (OpenKalman::layout_of_v<Arg> == OpenKalman::data_layout::none)
    struct Strides<Arg>
#else
    template<typename Arg>
  struct Strides<Arg, std::enable_if_t<(OpenKalman::layout_of_v<Arg> == OpenKalman::data_layout::none)>>
#endif
    { using type = std::tuple<std::size_t, std::size_t>; };

    using Stride0 = collections::collection_element_t<0, typename Strides<T>::type>;
    using Stride1 = collections::collection_element_t<1, typename Strides<T>::type>;

    using IndexType = typename std::decay_t<T>::Index;

    static constexpr IndexType stride0 = OpenKalman::values::fixed<Stride0, std::ptrdiff_t> ? static_cast<std::ptrdiff_t>(Stride0{}) : Eigen::Dynamic;
    static constexpr IndexType stride1 = OpenKalman::values::fixed<Stride1, std::ptrdiff_t> ? static_cast<std::ptrdiff_t>(Stride1{}) : Eigen::Dynamic;

  public:

    using XprKind = Eigen::MatrixXpr;
    using Scalar = OpenKalman::scalar_type_of_t<T>;
    using StorageKind = Eigen::Dense;
    using StorageIndex = Eigen::Index;
    enum {
      Flags = (layout_bit | direct | lvalue_bit) & ~Eigen::NestByRefBit,
      InnerStrideAtCompileTime = layout_bit == Eigen::RowMajorBit ? stride1 : stride0,
      OuterStrideAtCompileTime = layout_bit == Eigen::RowMajorBit ? stride0 : stride1,
      RowsAtCompileTime = OpenKalman::dynamic_dimension<T, 0> ? Eigen::Dynamic : static_cast<Eigen::Index>(OpenKalman::index_dimension_of_v<T, 0>),
      ColsAtCompileTime = OpenKalman::dynamic_dimension<T, 1> ? Eigen::Dynamic : static_cast<Eigen::Index>(OpenKalman::index_dimension_of_v<T, 1>),
      MaxRowsAtCompileTime [[maybe_unused]] = RowsAtCompileTime,
      MaxColsAtCompileTime [[maybe_unused]] = ColsAtCompileTime,
    };
  };

} // OpenKalman::Eigen3::internal


#endif