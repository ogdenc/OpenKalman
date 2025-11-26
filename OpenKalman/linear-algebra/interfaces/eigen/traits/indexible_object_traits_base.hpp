/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Base for type traits as applied to native Eigen types.
 */

#ifndef OPENKALMAN_OBJECT_TRAITS_BASE_HPP
#define OPENKALMAN_OBJECT_TRAITS_BASE_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<Eigen3::eigen_general T>
  struct object_traits_base<T>
#else
  template<typename T>
  struct object_traits_base<T, std::enable_if_t<Eigen3::eigen_general<T>>>
#endif
  {
  private:

    using IndexType = typename T::Index; // static_assert(values::index<IndexType>);

  public:

    using scalar_type = typename T::Scalar;


    template<typename Arg>
    static constexpr auto
    count_indices(const Arg& arg)
    {
      constexpr bool lin = (Eigen::internal::evaluator<Arg>::Flags & Eigen::LinearAccessBit) != 0x0;
      if constexpr (Arg::RowsAtCompileTime == 1 and Arg::ColsAtCompileTime == 1)
        return std::integral_constant<std::size_t, 0_uz>{};
      else if constexpr (Arg::ColsAtCompileTime == 1 and lin)
        return std::integral_constant<std::size_t, 1_uz>{};
      else
        return std::integral_constant<std::size_t, 2_uz>{};
    }


    template<typename Arg, typename N>
    static constexpr auto
    get_pattern_collection(const Arg& arg, N n)
    {
      if constexpr (values::fixed<N>)
      {
        constexpr auto dim = n == 0_uz ? Arg::RowsAtCompileTime : Arg::ColsAtCompileTime;

        if constexpr (dim == Eigen::Dynamic)
        {
          if constexpr (n == 0_uz) return static_cast<std::size_t>(arg.rows());
          else return static_cast<std::size_t>(arg.cols());
        }
        else return Dimensions<static_cast<std::size_t>(dim)>{};
      }
      else
      {
        if (n == 0_uz) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols()); // n == 1_uz
        // n >= 2 is precluded by the general get_pattern_collection function
      }
    }

  protected:

    static constexpr bool row_major = (Eigen::internal::traits<T>::Flags & Eigen::RowMajorBit) != 0x0;

    static constexpr bool direct_access = (Eigen::internal::traits<T>::Flags & Eigen::DirectAccessBit) != 0x0;

    static constexpr bool lvalue_access = (Eigen::internal::traits<T>::Flags & Eigen::LvalueBit) != 0x0;


  public:

    static constexpr bool is_writable = lvalue_access and direct_access;


#ifdef __cpp_lib_concepts
    template<typename Arg> requires requires(Arg&& arg) { {*std::forward<Arg>(arg).data()} -> values::scalar; } and direct_access
#else
    template<typename Arg, std::enable_if_t<values::scalar<decltype(*std::declval<Arg&&>().data())> and direct_access, int> = 0>
#endif
    static constexpr decltype(auto)
    raw_data(Arg&& arg) { return std::forward<Arg>(arg).data(); }

  private:

#ifdef __cpp_lib_concepts
    template<typename U>
#else
    template<typename U, typename = void>
#endif
    struct has_strides : std::false_type {};


#ifdef __cpp_lib_concepts
    template<typename U> requires requires(U u) { {u.innerStride()} -> values::index; {u.outerStride()} -> values::index; }
    struct has_strides<U>
#else
    template<typename U>
    struct has_strides<U, std::enable_if_t<values::index<decltype(std::declval<U>().innerStride())> and
      values::index<decltype(std::declval<U>().outerStride())>>>
#endif
      : std::true_type {};

  public:

    static constexpr data_layout layout = has_strides<T>::value and direct_access ? data_layout::stride : row_major ? data_layout::right : data_layout::left;


#ifdef __cpp_lib_concepts
    template<Eigen3::eigen_dense_general Arg> requires has_strides<T>::value
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_dense_general<Arg> and has_strides<T>::value, int> = 0>
#endif
    static auto
    strides(Arg&& arg)
    {
      constexpr auto outer = Eigen::internal::traits<T>::OuterStrideAtCompileTime;
      constexpr auto inner = Eigen::internal::traits<T>::InnerStrideAtCompileTime;
      if constexpr (outer != Eigen::Dynamic and inner != Eigen::Dynamic)
        return std::tuple {std::integral_constant<std::ptrdiff_t, outer>{}, std::integral_constant<std::ptrdiff_t, inner>{}};
      else if constexpr (outer != Eigen::Dynamic and inner == Eigen::Dynamic)
        return std::tuple {std::integral_constant<std::ptrdiff_t, outer>{}, arg.innerStride()};
      else if constexpr (outer == Eigen::Dynamic and inner != Eigen::Dynamic)
        return std::tuple {arg.outerStride(), std::integral_constant<std::ptrdiff_t, inner>{}};
      else
        return std::tuple {arg.outerStride(), arg.innerStride()};
    }

  };


}


#endif
