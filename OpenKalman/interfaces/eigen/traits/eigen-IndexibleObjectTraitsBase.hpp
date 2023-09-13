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

#ifndef OPENKALMAN_EIGEN_INDEXIBLEOBJECTTRAITSBASE_HPP
#define OPENKALMAN_EIGEN_INDEXIBLEOBJECTTRAITSBASE_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::Eigen3
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct eigen_has_linear_access : std::false_type {};

    template<typename T>
    struct eigen_has_linear_access<T, std::enable_if_t<Eigen3::eigen_dense_general<T, true>>>
      : std::bool_constant<(Eigen::internal::evaluator<T>::Flags & Eigen::LinearAccessBit) != 0> {};
  }
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_dense_general T>
  struct IndexibleObjectTraitsBase<T>
#else
  template<typename T>
  struct IndexibleObjectTraitsBase<T, std::enable_if_t<Eigen3::eigen_dense_general<T>>>
#endif
  {
    static constexpr std::size_t max_indices = 2;

    using index_type = typename std::decay_t<T>::Index;

    using scalar_type = typename std::decay_t<T>::Scalar;


    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        constexpr auto i = static_index_value_of_v<N>;
        constexpr index_type_of_t<Arg> dim = i == 0 ? std::decay_t<T>::RowsAtCompileTime : std::decay_t<T>::ColsAtCompileTime;

        if constexpr (dim == Eigen::Dynamic)
        {
          if constexpr (i == 0) return static_cast<std::size_t>(arg.rows());
          else return static_cast<std::size_t>(arg.cols());
        }
        else return Dimensions<dim>{};
      }
      else
      {
        if constexpr (n == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<index_type>...I> requires (sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0)
#else
    template<typename Arg, typename...I, std::enable_if_t<(sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (std::is_convertible_v<I, index_type> and ...) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0) , int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i)
    {
      if constexpr ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0)
        return std::forward<Arg>(arg).coeffRef(static_cast<index_type>(i)...);
      else
        return std::forward<Arg>(arg).coeff(static_cast<index_type>(i)...);
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<index_type>...I> requires (sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0) and
      ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0)
#else
    template<typename Arg, typename...I, std::enable_if_t<(sizeof...(I) > 0) and (sizeof...(I) <= 2) and
      (std::is_convertible_v<I, index_type> and ...) and
      (sizeof...(I) != 1 or (Eigen::internal::evaluator<std::decay_t<Arg>>::Flags & Eigen::LinearAccessBit) != 0) and
      ((std::decay_t<Arg>::Flags & Eigen::LvalueBit) != 0), int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
    {
      arg.coeffRef(i...) = s;
    }


    static constexpr bool is_writable =
      (Eigen::internal::traits<std::decay_t<T>>::Flags & (Eigen::LvalueBit | Eigen::DirectAccessBit)) != 0x0;


#ifdef __cpp_lib_concepts
    template<typename Arg> requires requires(Arg& arg) { requires std::is_pointer_v<decltype(arg.data())>; }
#else
    template<typename Arg, std::enable_if_t<std::is_pointer<decltype(std::declval<Arg&>().data())>::value, int> = 0>
#endif
    static constexpr auto*
    data(Arg& arg) { return arg.data(); }


    static constexpr Layout layout = (Eigen::internal::traits<T>::Flags & Eigen::RowMajorBit) != 0x0 ? Layout::right : Layout::left;

  };


  //---- DiagonalMatrix and DiagonalWrapper cases ----//


#ifdef __cpp_concepts
  template<typename T> requires Eigen3::eigen_DiagonalMatrix<T> or Eigen3::eigen_DiagonalWrapper<T>
  struct IndexibleObjectTraitsBase<T>
#else
  template<typename T>
  struct IndexibleObjectTraitsBase<T, std::enable_if_t<Eigen3::eigen_DiagonalMatrix<T> or Eigen3::eigen_DiagonalWrapper<T>>>
#endif
  {
    static constexpr std::size_t max_indices = 2;

    using index_type = typename std::decay_t<T>::Index;

    using scalar_type = typename std::decay_t<T>::Scalar;


#ifdef __cpp_concepts
    template<typename Arg> requires element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1>
#else
    template<typename Arg, std::enable_if_t<element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1>, int> = 0>
#endif
    static scalar_type_of_t<Arg> get(const Arg& arg, index_type i)
    {
      return get_element(nested_matrix(arg), i);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1> or
      element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 2>
#else
    template<typename Arg, std::enable_if_t<element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 1> or
      element_gettable<decltype(nested_matrix(std::declval<const Arg&>())), 2>, int> = 0>
#endif
    static constexpr auto get(const Arg& arg, index_type i, index_type j)
    {
      if (i == j)
      {
        if constexpr (element_gettable<nested_matrix_of_t<const Arg&>, 1>)
          return get_element(nested_matrix(arg), i);
        else
          return get_element(nested_matrix(arg), i, i);
      }
      else
      {
        return scalar_type_of_t<Arg>(0);
      }
    }


#ifdef __cpp_concepts
    template<typename Arg> requires element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<nested_matrix_of_t<Arg&>, 2>
#else
    template<typename Arg, std::enable_if_t<element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<typename nested_matrix_of<Arg&>::type, 2>, int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, index_type i)
    {
      if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
        set_element(nested_matrix(arg), s, i);
      else
        set_element(nested_matrix(arg), s, i, i);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<decltype(nested_matrix(std::declval<Arg&>())), 2>
#else
    template<typename Arg, std::enable_if_t<element_settable<decltype(nested_matrix(std::declval<Arg&>())), 1> or
      element_settable<decltype(nested_matrix(std::declval<Arg&>())), 2>, int> = 0>
#endif
    static void set(Arg& arg, const scalar_type_of_t<Arg>& s, index_type i, index_type j)
    {
      if (i == j or s == 0)
      {
        if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, i);
      }
      else throw std::invalid_argument {"Off-diagonal elements of DiagonalMatrix or DiagonalWrapper can only be set to zero"};
    }

    static constexpr bool is_writable = false;

  };

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN_INDEXIBLEOBJECTTRAITSBASE_HPP
