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
 * \brief Overloaded general linear-algebra functions.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Take the conjugate of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto) conjugate(Arg&& arg) noexcept
  {
    if constexpr (not complex_number<scalar_type_of_t<Arg>> or zero_matrix<Arg> or identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr auto c = constant_coefficient_v<Arg>;
      if constexpr (imaginary_part(c) == 0)
        return std::forward<Arg>(arg);
      else
# if __cpp_nontype_template_args >= 201911L
        return make_constant_matrix_like<conjugate(c)>(std::forward<Arg>(arg));
# else
        return make_self_contained(c * make_constant_matrix_like<1>(std::forward<Arg>(arg)));
# endif
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      constexpr auto c = constant_diagonal_coefficient_v<Arg>;
      if constexpr (imaginary_part(c) == 0)
        return std::forward<Arg>(arg);
      else
# if __cpp_nontype_template_args >= 201911L
        return to_diagonal(make_constant_matrix_like<conjugate(c)>(diagonal_of(std::forward<Arg>(arg))));
# else
        return make_self_contained(c * to_diagonal(make_constant_matrix_like<1>(diagonal_of(std::forward<Arg>(arg)))));
# endif
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return to_diagonal(conjugate(diagonal_of(std::forward<Arg>(arg))));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
  }


  namespace detail
  {
    template<typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_zero(Arg&& arg, std::index_sequence<Is...>) noexcept
    {
      return make_zero_matrix_like<Arg>(get_dimensions_of<1>(arg), get_dimensions_of<0>(arg), get_dimensions_of<Is + 2>(arg)...);
    }

    template<auto c, typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_constant(Arg&& arg, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<Arg, c>(get_dimensions_of<1>(arg), get_dimensions_of<0>(arg), get_dimensions_of<Is + 2>(arg)...);
    }
  }


  /**
   * \brief Take the transpose of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto) transpose(Arg&& arg) noexcept
  {
    if constexpr (diagonal_matrix<Arg> or (hermitian_matrix<Arg> and not complex_number<scalar_type_of_t<Arg>>) or
      (constant_matrix<Arg> and square_matrix<Arg>))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      constexpr std::size_t dims = std::max({max_indices_of_v<Arg>, static_cast<std::size_t>(2)});
      return detail::transpose_zero(std::forward<Arg>(arg), std::make_index_sequence<dims - 2> {});
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr std::make_index_sequence<std::max({max_indices_of_v<Arg>, static_cast<std::size_t>(2)}) - 2> seq;
# if __cpp_nontype_template_args >= 201911L
      return detail::transpose_constant<constant_coefficient_v<Arg>>(std::forward<Arg>(arg), seq);
# else
      constexpr auto c = real_projection(constant_coefficient_v<Arg>);
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<decltype(c)>(c_integral)))
        return detail::transpose_constant<c_integral>(std::forward<Arg>(arg), seq);
      else
        return make_self_contained(c * detail::transpose_constant<1>(std::forward<Arg>(arg), seq));
# endif
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the adjoint of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  constexpr decltype(auto) adjoint(Arg&& arg) noexcept
  {
    if constexpr (hermitian_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg> or not complex_number<scalar_type_of_t<Arg>>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr auto c = constant_coefficient_v<Arg>;
      if constexpr (imaginary_part(c) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
      {
        constexpr std::make_index_sequence<std::max({max_indices_of_v<Arg>, static_cast<std::size_t>(2)}) - 2> seq;
        return detail::transpose_constant<conjugate(c)>(std::forward<Arg>(arg), seq);
      }
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the determinant of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg>
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe>, int> = 0>
#endif
  constexpr auto determinant(Arg&& arg)
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In determinant, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return real_projection(Scalar(1));
    }
    else if constexpr (zero_matrix<Arg> or (constant_matrix<Arg> and not one_by_one_matrix<Arg>))
    {
      return real_projection(Scalar(0));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient_v<Arg>; //< One-by-one case. General case is handled above.
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
        return std::pow(constant_diagonal_coefficient_v<Arg>, get_index_dimension_of<0>(arg));
      else
        return internal::constexpr_pow(constant_diagonal_coefficient_v<Arg>, row_dimension_of_v<Arg>);
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else if constexpr (triangular_matrix<Arg>) // this includes diagonal case
    {
      return reduce(std::multiplies<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      if constexpr (hermitian_matrix<Arg> and complex_number<std::decay_t<decltype(r)>>) return real_projection(r);
      else return r;
    }
  }


#ifdef __cpp_concepts
  /**
   * \brief Take the trace of a matrix
   * \tparam Arg The matrix
   */
  template<square_matrix<Likelihood::maybe> Arg>
#else
  template<typename Arg, std::enable_if_t<(square_matrix<Arg, Likelihood::maybe>), int> = 0>
#endif
  constexpr auto trace(Arg&& arg)
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In trace, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(get_index_dimension_of<0>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return Scalar(constant_coefficient_v<Arg> * get_index_dimension_of<0>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return Scalar(constant_diagonal_coefficient_v<Arg> * get_index_dimension_of<0>(arg));
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(std::forward<Arg>(arg), std::size_t(0), std::size_t(0));
    }
    else
    {
      return reduce(std::plus<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
  }


  namespace detail
  {
    template<typename A, typename B, std::size_t...Is>
    static constexpr decltype(auto) contract_zero(A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return make_zero_matrix_like<A>(
        get_dimensions_of<0>(a), get_dimensions_of<1>(b), get_dimensions_of<Is + 2>(a)...);
    }

    template<auto c, typename A, typename B, std::size_t...Is>
    static constexpr decltype(auto) contract_constant(A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<A, c>(get_dimensions_of<0>(a), get_dimensions_of<1>(b), get_dimensions_of<Is + 2>(a)...);
    }


    template<std::size_t I, typename T, typename...Ts>
    constexpr decltype(auto) best_descriptor(T&& t, Ts&&...ts)
    {
       if constexpr (sizeof...(Ts) == 0 or dynamic_dimension<T, I>) return get_dimensions_of<I>(t);
       else return best_descriptor<I>(std::forward<Ts>(ts)...);
    }


    template<std::size_t...I, typename T>
    constexpr decltype(auto) sum_impl(std::index_sequence<I...>, T&& t) { return std::forward<T>(t); }


    template<std::size_t...I, typename T0, typename T1, typename...Ts>
    constexpr decltype(auto) sum_impl(std::index_sequence<I...> seq, T0&& t0, T1&& t1, Ts&&...ts)
    {
      using Scalar = std::common_type_t<scalar_type_of_t<T0>, scalar_type_of_t<T1>>;

      if constexpr ((constant_matrix<T0> or constant_matrix<T1>) and not index_descriptors_match<T0, T1>)
      {
        if (not get_index_descriptors_match(t0, t1))
          throw std::invalid_argument {"In sum function, index descriptors of arguments do not match"};
      }

      if constexpr (zero_matrix<T0>)
      {
        return sum_impl(seq, std::forward<T1>(t1), std::forward<Ts>(ts)...);
      }
      else if constexpr (zero_matrix<T1>)
      {
        return sum_impl(seq, std::forward<T0>(t0), std::forward<Ts>(ts)...);
      }
      else if constexpr ((constant_matrix<T0> and constant_matrix<T1>))
      {
        constexpr auto c = constant_coefficient_v<T0> + constant_coefficient_v<T1>;
# if __cpp_nontype_template_args >= 201911L
        return sum_impl(seq, make_constant_matrix_like<T0, c, Scalar>(best_descriptor<I>(t0, t1, ts...)...), std::forward<Ts>(ts)...);
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
          return sum_impl(seq, make_constant_matrix_like<T0, c_integral, Scalar>(best_descriptor<I>(t0, t1, ts...)...), std::forward<Ts>(ts)...);
        else
          return make_self_contained(c * sum_impl(seq, make_constant_matrix_like<T0, 1, Scalar>(best_descriptor<I>(t0, t1, ts...)...), std::forward<Ts>(ts)...));
# endif
      }
      else if constexpr (constant_matrix<T0> and sizeof...(Ts) > 0)
      {
        return sum_impl(seq, std::forward<T1>(t1), sum_impl(seq, std::forward<T0>(t0), std::forward<Ts>(ts)...));
      }
      else if constexpr (constant_matrix<T1> and sizeof...(Ts) > 0)
      {
        return sum_impl(seq, std::forward<T0>(t0), sum_impl(seq, std::forward<T1>(t1), std::forward<Ts>(ts)...));
      }
      else if constexpr (diagonal_matrix<T0> and diagonal_matrix<T1>)
      {
        auto d0 = to_diagonal(sum_impl(seq, diagonal_of(std::forward<T0>(t0)), diagonal_of(std::forward<T1>(t1))));
        auto ret = sum_impl(seq, std::move(d0), std::forward<Ts>(ts)...);
        return ret;
      }
      else
      {
        auto ret = sum_impl(seq, interface::LinearAlgebra<std::decay_t<T0>>::sum(std::forward<T0>(t0), std::forward<T1>(t1)), std::forward<Ts>(ts)...);
        return ret;
      }
    }
  } // namespace detail


  /**
   * \brief Element-by-element sum of one or more objects.
   */
#ifdef __cpp_concepts
  template<indexible...Ts> requires (sizeof...(Ts) > 0) and maybe_index_descriptors_match<Ts...>
#else
  template<typename...Ts, std::enable_if_t<(indexible<Ts> and ...) and (sizeof...(Ts) > 0) and
    maybe_index_descriptors_match<Ts...>, int> = 0>
#endif
  constexpr decltype(auto) sum(Ts&&...ts)
  {
    constexpr std::make_index_sequence<std::max({max_indices_of_v<Ts>...})> seq;
    return detail::sum_impl(seq, std::forward<Ts>(ts)...);
  }


  /* // Only for use with alternate code below
  namespace detail
  {
    template<typename A, typename B, std::size_t...Is>
    static constexpr auto contract_dimensions(A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return std::tuple {get_dimensions_of<0>(a), get_dimensions_of<1>(b), get_dimensions_of<Is + 2>(a)...};
    }
  }*/


  /**
   * \brief Matrix multiplication of A * B.
   */
#ifdef __cpp_concepts
  template<indexible A, indexible B> requires dynamic_dimension<A, 1> or dynamic_dimension<B, 0> or
    (index_dimension_of_v<A, 1> == index_dimension_of_v<B, 0>)
#else
  template<typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (dynamic_dimension<A, 1> or dynamic_dimension<B, 0> or (index_dimension_of_v<A, 1> == index_dimension_of_v<B, 0>)), int> = 0>
#endif
  constexpr decltype(auto) contract(A&& a, B&& b)
  {
    if constexpr (dynamic_dimension<A, 1> or dynamic_dimension<B, 0>) if (get_dimensions_of<1>(a) != get_dimensions_of<0>(b))
      throw std::domain_error {"In contract, columns of a (" + std::to_string(get_index_dimension_of<1>(a)) +
        ") do not match rows of b (" + std::to_string(get_index_dimension_of<0>(b)) + ")"};

    constexpr std::size_t dims = std::max({max_indices_of_v<A>, max_indices_of_v<B>, static_cast<std::size_t>(2)});
    constexpr std::make_index_sequence<dims - 2> seq;

    if constexpr (identity_matrix<B>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (identity_matrix<A>)
    {
      return std::forward<B>(b);
    }
    else if constexpr (zero_matrix<A> or zero_matrix<B>)
    {
      return detail::contract_zero(std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (constant_matrix<A> and constant_matrix<B>)
    {
      constexpr auto c = constant_coefficient_v<A> * constant_coefficient_v<B>;
      if constexpr (dynamic_dimension<A, 1> and dynamic_dimension<B, 0>)
      {
        auto r = get_index_dimension_of<1>(a);
        return r * detail::contract_constant<c>(std::forward<A>(a), std::forward<B>(b), seq);
      }
      else
      {
        constexpr auto r = dynamic_dimension<A, 1> ? index_dimension_of_v<B, 0> : index_dimension_of_v<A, 1>;
# if __cpp_nontype_template_args >= 201911L
        return detail::contract_constant<c * r>(std::forward<A>(a), std::forward<B>(b), seq);
# else
        constexpr auto cr_integral = static_cast<std::intmax_t>(c * r);
        if constexpr (are_within_tolerance(c * r, static_cast<scalar_type_of_t<A>>(cr_integral)))
          return detail::contract_constant<cr_integral>(std::forward<A>(a), std::forward<B>(b), seq);
        else
          return make_self_contained(c * detail::contract_constant<r>(std::forward<A>(a), std::forward<B>(b), seq));
# endif
      }
    }
    else if constexpr (diagonal_matrix<A> and constant_matrix<B>)
    {
      auto col = make_self_contained(diagonal_of(std::forward<A>(a)) * constant_coefficient_v<B>);
      return chipwise_operation<1>([&]{ return col; }, get_index_dimension_of<1>(b));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(col)));
    }
    else if constexpr (constant_matrix<A> and diagonal_matrix<B>)
    {
      auto row = make_self_contained(transpose(diagonal_of(std::forward<B>(b))) * constant_coefficient_v<A>);
      return chipwise_operation<0>([&]{ return row; }, get_index_dimension_of<0>(a));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(row)));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<B>)
    {
      auto ret = to_diagonal(n_ary_operation(std::multiplies<scalar_type_of_t<A>>{}, diagonal_of(std::forward<A>(a)), diagonal_of(std::forward<B>(b))));
      return ret;
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::contract(std::forward<A>(a), std::forward<B>(b));
    }
  }


  /**
   * \brief In-place matrix multiplication of A * B, storing the result in A
   * \tparam on_the_right Whether the application is on the right (true) or on the left (false)
   * \result Either either A * B (if on_the_right == true) or B * A (if on_the_right == false)
   */
#ifdef __cpp_concepts
  template<bool on_the_right = true, writable A, indexible B>
  requires (dynamic_dimension<A, 0> or dynamic_dimension<A, 1> or index_dimension_of_v<A, 0> == index_dimension_of_v<A, 1>) and
    (dynamic_dimension<A, 0> or dynamic_dimension<B, 0> or index_dimension_of_v<A, 0> == index_dimension_of_v<B, 0>) and
    (dynamic_dimension<A, 1> or dynamic_dimension<B, 1> or index_dimension_of_v<A, 1> == index_dimension_of_v<B, 1>)
#else
  template<bool on_the_right = true, typename A, typename B, std::enable_if_t<writable<A> and indexible<B> and
    (dynamic_dimension<A, 0> or dynamic_dimension<A, 1> or index_dimension_of_v<A, 0> == index_dimension_of_v<A, 1>) and
    (dynamic_dimension<A, 0> or dynamic_dimension<B, 0> or index_dimension_of_v<A, 0> == index_dimension_of_v<B, 0>) and
    (dynamic_dimension<A, 1> or dynamic_dimension<B, 1> or index_dimension_of_v<A, 1> == index_dimension_of_v<B, 1>), int> = 0>
#endif
  constexpr A& contract_in_place(A& a, B&& b)
  {
    if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<A, 1>) if (get_dimensions_of<0>(a) != get_dimensions_of<1>(a))
      throw std::domain_error {"In contract_in_place, argument a is not square"};

    if constexpr (dynamic_dimension<A, 0> or dynamic_dimension<B, 0>) if (get_dimensions_of<0>(a) != get_dimensions_of<0>(b))
      throw std::domain_error {"In contract_in_place, rows of a (" + std::to_string(get_index_dimension_of<0>(a)) +
        ") do not match rows of b (" + std::to_string(get_index_dimension_of<0>(b)) + ")"};

    if constexpr (dynamic_dimension<A, 1> or dynamic_dimension<B, 1>) if (get_dimensions_of<1>(a) != get_dimensions_of<1>(b))
      throw std::domain_error {"In contract_in_place, columns of a (" + std::to_string(get_index_dimension_of<1>(a)) +
        ") do not match columns of b (" + std::to_string(get_index_dimension_of<1>(b)) + ")"};

    if constexpr (identity_matrix<B>)
    {
      return a;
    }
    else if constexpr (zero_matrix<B>)
    {
      return a = std::forward<B>(b);
    }
    //else if constexpr (diagonal_matrix<A> and diagonal_matrix<B>)
    //{
    //  \todo Create and a new set_diagonal function for this (similar to set_chip).
    //  set_diagonal(n_ary_operation(std::multiplies<scalar_type_of_t<A>>{}, diagonal_of(a), diagonal_of(std::forward<B>(b))));
    //  return a;
    //}
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
