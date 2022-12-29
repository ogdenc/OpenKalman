/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief \ref reduce and related functions for reductions.
 */

#ifndef OPENKALMAN_REDUCTION_FUNCTIONS_HPP
#define OPENKALMAN_REDUCTION_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // -------- //
  //  reduce  //
  // -------- //

  namespace detail
  {
    template<std::size_t I, std::size_t...index, typename Arg>
    constexpr auto get_reduced_index(Arg&& arg)
    {
      if constexpr (((I == index) or ...))
      {
        using T = coefficient_types_of_t<Arg, I>;
        if constexpr (has_uniform_dimension_type<T>) return uniform_dimension_type_of_t<T>{};
        else return Dimensions<1>{};
      }
      else return get_dimensions_of<I>(std::forward<Arg>(arg));
    }


    template<std::size_t...index, typename T, std::size_t...I>
    constexpr auto make_zero_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      return make_zero_matrix_like<T>(get_reduced_index<I, index...>(std::forward<T>(t))...);
    }


    template<auto constant, std::size_t...index, typename T, std::size_t...I>
    constexpr auto make_constant_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      return make_constant_matrix_like<T, constant>(get_reduced_index<I, index...>(std::forward<T>(t))...);
    }


    template<auto constant, std::size_t index, typename T, std::size_t...I>
    constexpr auto make_constant_diagonal_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      // \todo Handle 3+ dimensional constant diagonal tensors
      static_assert(index == 0 or index == 1);
      return make_constant_matrix_like<T, constant>(get_reduced_index<I, index>(std::forward<T>(t))...);
    }


    template<typename T, std::size_t...indices, std::size_t...Is>
    constexpr std::size_t count_reduced_dimensions(std::index_sequence<Is...>)
    {
      return ([]{
        constexpr auto I = Is;
        if constexpr (((I == indices) or ...)) return index_dimension_of_v<T, I>;
        else return 1;
      }() * ... * 1);
    }


    template<std::size_t...indices, typename T, std::size_t...Is>
    constexpr std::size_t count_reduced_dimensions(const T& t, std::index_sequence<Is...>)
    {
      return ([](const T& t){
        constexpr auto I = Is;
        if constexpr (((I == indices) or ...)) return get_index_dimension_of<I>(t);
        else return 1;
      }(t) * ... * 1);
    }


    template<std::size_t dim, typename BinaryFunction, typename Scalar>
    constexpr Scalar calc_reduce_constant(Scalar constant)
    {
      if constexpr (dim <= 1)
        return constant;
      else if constexpr (internal::is_plus<BinaryFunction>::value)
        return constant * static_cast<Scalar>(dim);
      else if constexpr (internal::is_multiplies<BinaryFunction>::value)
        return internal::constexpr_pow(constant, dim);
      else
        return BinaryFunction{}(constant, calc_reduce_constant<dim - 1, BinaryFunction>(constant));
    }


    template<typename BinaryFunction, typename Scalar>
    constexpr Scalar calc_reduce_constant(std::size_t dim, const BinaryFunction& b, Scalar constant)
    {
      if (dim <= 1)
        return constant;
      else if constexpr (internal::is_plus<BinaryFunction>::value)
        return constant * dim;
      else if constexpr (internal::is_multiplies<BinaryFunction>::value)
        return std::pow(constant, dim);
      else
        return b(constant, calc_reduce_constant(dim - 1, b, constant));
    }


    template<typename Arg, std::size_t...I>
    constexpr bool has_uniform_reduction_indices(std::index_sequence<I...>)
    {
      return ((has_uniform_dimension_type<coefficient_types_of_t<Arg, I>> or dynamic_dimension<Arg, I>) and ...);
    }


    template<typename BinaryFunction, typename Arg, std::size_t...indices>
    constexpr scalar_type_of_t<Arg>
    reduce_all_indices(const BinaryFunction& b, Arg&& arg, std::index_sequence<indices...>)
    {
      if constexpr (zero_matrix<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
      {
        return 0;
      }
      else if constexpr (constant_matrix<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        constexpr Scalar c = constant_coefficient_v<Arg>;
        constexpr auto seq = std::make_index_sequence<max_indices_of_v<Arg>> {};
        constexpr bool fixed_reduction_dims = not (dynamic_dimension<Arg, indices> or ...);

        if constexpr (fixed_reduction_dims and internal::constexpr_n_ary_function<BinaryFunction, Arg, Arg>)
        {
          constexpr std::size_t dim = count_reduced_dimensions<Arg, indices...>(seq);
          return calc_reduce_constant<dim, BinaryFunction>(c);
        }
        else
        {
          std::size_t dim = count_reduced_dimensions<indices...>(arg, seq);
          return calc_reduce_constant(dim, b, c);
        }
      }
      else
      {
        decltype(auto) red = interface::ArrayOperations<std::decay_t<Arg>>::template reduce<indices...>(b, std::forward<Arg>(arg));
        using Red = decltype(red);

        static_assert(scalar_type<Red> or one_by_one_matrix<Red, Likelihood::maybe>);

        if constexpr (scalar_type<Red>)
          return std::forward<Red>(red);
        else if constexpr (element_gettable<Red, decltype(indices)...>)
          return get_element(std::forward<Red>(red), static_cast<decltype(indices)>(0)...);
        else
          return interface::LinearAlgebra<std::decay_t<Red>>::trace(std::forward<Red>(red));
      }
    }

  } // namespace detail


  /**
   * \brief Perform a partial reduction based on an associative binary function, across one or more indices.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
    indexible<Arg> and ((index < max_indices_of<Arg>::value) and ... and (indices < max_indices_of<Arg>::value)) and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(reduce<index, indices...>(b, nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (index_dimension_of_v<Arg, index> == 1)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return reduce<indices...>(b, std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
    {
      return detail::make_zero_matrix_reduction<index, indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      using Scalar = scalar_type_of_t<Arg>;
      constexpr Scalar c_arg = constant_coefficient_v<Arg>;
      constexpr bool fixed_reduction_dims = not (dynamic_dimension<Arg, index> or ... or dynamic_dimension<Arg, indices>);

      if constexpr (fixed_reduction_dims and internal::constexpr_n_ary_function<BinaryFunction, Arg, Arg>)
      {
        constexpr std::size_t dim = detail::count_reduced_dimensions<Arg, index, indices...>(seq);
        constexpr auto c = detail::calc_reduce_constant<dim, BinaryFunction>(c_arg);
# if __cpp_nontype_template_args >= 201911L
        return detail::make_constant_matrix_reduction<c, index, indices...>(std::forward<Arg>(arg), seq);
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
          return detail::make_constant_matrix_reduction<c_integral, index, indices...>(std::forward<Arg>(arg), seq);
        else
          return make_self_contained(c * detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq));
# endif
      }
      else
      {
        std::size_t dim = detail::count_reduced_dimensions<index, indices...>(arg, seq);
        auto c = detail::calc_reduce_constant(dim, b, c_arg);
        auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
        return make_self_contained(c * to_native_matrix<Arg>(std::move(red)));
      }
    }
    // \todo Add this after updating DiagonalMatrix to include diagonals of any rank
    //else if constexpr (constant_diagonal_matrix<Arg>)
    //{
    //}
    else
    {
      return interface::ArrayOperations<std::decay_t<Arg>>::template reduce<index, indices...>(b, std::forward<Arg>(arg));
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction based on an associative binary function, and return a scalar.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A scalar representing a complete reduction.
   */
#ifdef __cpp_concepts
  template<typename BinaryFunction, indexible Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {})), int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    return detail::reduce_all_indices(b, std::forward<Arg>(arg), std::make_index_sequence<max_indices> {});
  }


  // ---------------- //
  //  average_reduce  //
  // ---------------- //

  /**
   * \brief Perform a partial reduction by taking the average along one or more indices.
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename Arg, std::enable_if_t<indexible<Arg> and
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  average_reduce(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(average_reduce<index, indices...>(nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (index_dimension_of_v<Arg, index> == 1)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return average_reduce<indices...>(std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return detail::make_zero_matrix_reduction<index, indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr Scalar c = constant_coefficient_v<Arg>;
# if __cpp_nontype_template_args >= 201911L
      return detail::make_constant_matrix_reduction<c, index, indices...>(std::forward<Arg>(arg), seq);
# else
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
      {
        return detail::make_constant_matrix_reduction<c_integral, index, indices...>(std::forward<Arg>(arg), seq);
      }
      else
      {
        auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
        return make_self_contained(c * to_native_matrix<Arg>(red));
      }
# endif
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (not dynamic_dimension<Arg, 0>)
      {
        constexpr auto c = static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>) / index_dimension_of_v<Arg, 0>;
# if __cpp_nontype_template_args >= 201911L
        auto ret = detail::make_constant_diagonal_matrix_reduction<c, index>(std::forward<Arg>(arg), seq);
        if constexpr (sizeof...(indices) > 0) return average_reduce<indices...>(std::move(ret));
        else return ret;
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        {
          auto ret = detail::make_constant_diagonal_matrix_reduction<c_integral, index>(std::forward<Arg>(arg), seq);
          if constexpr (sizeof...(indices) > 0) return average_reduce<indices...>(std::move(ret));
          else return ret;
        }
        else
        {
          auto ret = detail::make_constant_diagonal_matrix_reduction<1, index>(std::forward<Arg>(arg), seq);
          if constexpr (sizeof...(indices) > 0)
            return make_self_contained(c * average_reduce<indices...>(std::move(ret)));
          else
            return make_self_contained(c * std::move(ret));
        }
# endif
      }
      else
      {
        auto c = static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>) / get_index_dimension_of<0>(arg);
        auto ret = detail::make_constant_diagonal_matrix_reduction<1, index>(std::forward<Arg>(arg), seq);
        if constexpr (sizeof...(indices) > 0)
          return make_self_contained(c * average_reduce<indices...>(std::move(ret)));
        else
          return make_self_contained(c * std::move(ret));
      }
    }
    else
    {
      return make_self_contained(reduce<index, indices...>(std::plus<Scalar> {}, std::forward<Arg>(arg)) /
        (get_index_dimension_of<index>(arg) * ... * get_index_dimension_of<indices>(arg)));
    }
  }


  namespace detail
  {
    template<typename Arg, std::size_t I, std::size_t...Is>
    constexpr std::size_t const_diagonal_matrix_dim(const Arg& arg, std::index_sequence<I, Is...>)
    {
      if constexpr (sizeof...(Is) == 0) return get_index_dimension_of<I>(arg);
      else if constexpr(not dynamic_dimension<Arg, I>) return index_dimension_of_v<Arg, I>;
      else return const_diagonal_matrix_dim(arg, std::index_sequence<Is...>{});
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction by taking the average along all indices and returning a scalar value.
   * \returns A scalar representing the average of all components.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and
    detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}), int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  average_reduce(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
      return 0;
    else if constexpr (constant_matrix<Arg>)
      return constant_coefficient_v<Arg>;
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return static_cast<scalar_type_of_t<Arg>>(constant_diagonal_coefficient_v<Arg>) /
        detail::const_diagonal_matrix_dim(arg, std::make_index_sequence<max_indices_of_v<Arg>>{});
    }
    else
    {
      auto r = reduce(std::plus<scalar_type_of_t<Arg>> {}, std::forward<Arg>(arg));
      return r / (std::apply([](const auto&...d){ return (get_dimension_size_of(d) * ...); }, get_all_dimensions_of(arg)));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_REDUCTION_FUNCTIONS_HPP
