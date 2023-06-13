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

#include<complex>


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
      if constexpr (real_axis_number<constant_coefficient<Arg>>)
        return std::forward<Arg>(arg);
      else
        return make_constant_matrix_like(internal::scalar_constant_conj(constant_coefficient{arg}), std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (real_axis_number<constant_diagonal_coefficient<Arg>>)
        return std::forward<Arg>(arg);
      else
        return to_diagonal(make_constant_matrix_like(internal::scalar_constant_conj(constant_diagonal_coefficient{arg}),
          diagonal_of(std::forward<Arg>(arg))));
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
    template<typename C, typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_constant(C&& c, Arg&& arg, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<Arg>(std::forward<C>(c),
        get_dimensions_of<1>(arg), get_dimensions_of<0>(arg), get_dimensions_of<Is + 2>(arg)...);
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
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr std::make_index_sequence<std::max({max_indices_of_v<Arg>, static_cast<std::size_t>(2)}) - 2> seq;
      return detail::transpose_constant(constant_coefficient{arg}, std::forward<Arg>(arg), seq);
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
    else if constexpr (zero_matrix<Arg>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (real_axis_number<constant_coefficient<Arg>>)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not has_dynamic_dimensions<Arg> and row_dimension_of_v<Arg> == column_dimension_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
      {
        constexpr std::make_index_sequence<std::max({max_indices_of_v<Arg>, static_cast<std::size_t>(2)}) - 2> seq;
        return detail::transpose_constant(internal::scalar_constant_conj(constant_coefficient{arg}), std::forward<Arg>(arg), seq);
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
  constexpr auto determinant(Arg&& arg) -> scalar_type_of_t<Arg>
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In determinant, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    if constexpr (identity_matrix<Arg>)
    {
      return 1;
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (one_by_one_matrix<Arg>) return get_scalar_constant_value(constant_coefficient {arg});
      else return 0;
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return internal::scalar_constant_pow(constant_diagonal_coefficient{arg}, internal::index_dimension_scalar_constant_of<0>(arg))();
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg&&, 2>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else if constexpr (triangular_matrix<Arg>) // this includes diagonal case
    {
      return reduce(std::multiplies<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
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
  constexpr auto trace(Arg&& arg) -> scalar_type_of_t<Arg>
  {
    if constexpr (has_dynamic_dimensions<Arg>) if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
      throw std::domain_error {
        "In trace, rows of arg (" + std::to_string(get_index_dimension_of<0>(arg)) + ") do not match columns of arg (" +
        std::to_string(get_index_dimension_of<1>(arg)) + ")"};

    if constexpr (identity_matrix<Arg>)
    {
      return get_index_dimension_of<0>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return 0;
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return internal::scalar_constant_operation {std::multiplies<>{}, constant_coefficient{arg},
        internal::index_dimension_scalar_constant_of<0>(arg)}();
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return internal::scalar_constant_operation {std::multiplies<>{}, constant_diagonal_coefficient{arg},
        internal::index_dimension_scalar_constant_of<0>(arg)}();
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg&&, 2>)
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
    template<typename C, typename A, typename B, std::size_t...Is>
    static constexpr decltype(auto) contract_constant(C&& c, A&& a, B&& b, std::index_sequence<Is...>) noexcept
    {
      return make_constant_matrix_like<A>(std::forward<C>(c),
        get_dimensions_of<0>(a), get_dimensions_of<1>(b), get_dimensions_of<Is + 2>(a)...);
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
      if constexpr ((zero_matrix<T0> or zero_matrix<T1> or (constant_matrix<T0> and constant_matrix<T1>)) and not index_descriptors_match<T0, T1>)
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
        internal::scalar_constant_operation c {std::plus<>{}, constant_coefficient{t0}, constant_coefficient{t1}};
        auto cm = make_constant_matrix_like<T0>(std::move(c), best_descriptor<I>(t0, t1, ts...)...);
        return sum_impl(seq, std::move(cm), std::forward<Ts>(ts)...);
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
  template<indexible A, indexible B> requires dimension_size_of_index_is<A, 1, index_dimension_of_v<B, 0>, Likelihood::maybe>
#else
  template<typename A, typename B, std::enable_if_t<indexible<A> and indexible<B> and
    (dimension_size_of_index_is<A, 1, index_dimension_of<B, 0>::value, Likelihood::maybe>), int> = 0>
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
      using Scalar = std::decay_t<decltype(std::declval<scalar_type_of_t<A>>() * std::declval<scalar_type_of_t<B>>())>;
      return detail::contract_constant(internal::ScalarConstant<Likelihood::definitely, Scalar, 0>{}, std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (constant_matrix<A> and constant_matrix<B>)
    {
      internal::scalar_constant_operation ab {std::multiplies<>{}, constant_coefficient{a}, constant_coefficient{b}};

      auto dim_const = [](const auto& a, const auto& b) {
        if constexpr (dynamic_dimension<A, 1>) return internal::index_dimension_scalar_constant_of<0>(b);
        else return internal::index_dimension_scalar_constant_of<1>(a);
      }(a, b);

      internal::scalar_constant_operation abd {std::multiplies<>{}, std::move(ab), std::move(dim_const)};
      return detail::contract_constant(std::move(abd), std::forward<A>(a), std::forward<B>(b), seq);
    }
    else if constexpr (diagonal_matrix<A> and constant_matrix<B>)
    {
      auto col = make_self_contained(diagonal_of(std::forward<A>(a)) * constant_coefficient{b}());
      return chipwise_operation<1>([&]{ return col; }, get_index_dimension_of<1>(b));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(col)));
    }
    else if constexpr (constant_matrix<A> and diagonal_matrix<B>)
    {
      auto row = make_self_contained(transpose(diagonal_of(std::forward<B>(b))) * constant_coefficient{a}());
      return chipwise_operation<0>([&]{ return row; }, get_index_dimension_of<0>(a));
      //Another way to do this:
      //auto tup = detail::contract_dimensions(std::forward<A>(a), std::forward<B>(b), seq);
      //auto op = [](auto&& x){ return std::forward<decltype(x)>(x); };
      //return n_ary_operation(std::move(tup), op, make_self_contained(std::move(row)));
    }
    else if constexpr (diagonal_matrix<A> and diagonal_matrix<B>)
    {
      auto ret = to_diagonal(n_ary_operation(std::multiplies<>{}, diagonal_of(std::forward<A>(a)), diagonal_of(std::forward<B>(b))));
      return ret;
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<A>>::contract(std::forward<A>(a), std::forward<B>(b));
    }
  }


#ifndef __cpp_concepts
    namespace detail
    {
      template<bool on_the_right, typename A, typename B, typename = void>
      struct contract_in_place_exists : std::false_type {};

      template<bool on_the_right, typename A, typename B>
      struct contract_in_place_exists<on_the_right, A, B, std::void_t<decltype(
        interface::LinearAlgebra<std::decay_t<A>>::template contract<on_the_right>(std::declval<A&>(), std::declval<B&&>()))>>
        : std::true_type {};
    }
#endif


  /**
   * \brief In-place matrix multiplication of A * B, storing the result in A
   * \tparam on_the_right Whether the application is on the right (true) or on the left (false)
   * \result Either either A * B (if on_the_right == true) or B * A (if on_the_right == false)
   */
#ifdef __cpp_concepts
  template<bool on_the_right = true, square_matrix<Likelihood::maybe> A, square_matrix<Likelihood::maybe> B> requires
    maybe_has_same_shape_as<A, B> and (writable<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>)
#else
  template<bool on_the_right = true, typename A, typename B, std::enable_if_t<
    square_matrix<A, Likelihood::maybe> and square_matrix<B, Likelihood::maybe> and maybe_has_same_shape_as<A, B> and
    (writable<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>), int> = 0>
#endif
  constexpr A& contract_in_place(A& a, B&& b)
  {
    if constexpr (not square_matrix<A> or not square_matrix<B> or not has_same_shape_as<A, B>) if (not get_index_descriptors_match(a, b))
      throw std::invalid_argument {"Arguments to contract_in_place must match in size and be square matrices"};

    if constexpr (zero_matrix<A> or identity_matrix<B>)
    {
      return a;
    }
    else if constexpr (zero_matrix<B>)
    {
      return a = std::forward<B>(b);
    }
    else if constexpr (diagonal_adapter<A> and diagonal_matrix<B>)
    {
      internal::set_triangle<TriangleType::diagonal>(a, n_ary_operation(std::multiplies<>{}, diagonal_of(a), diagonal_of(std::forward<B>(b))));
      return a;
    }
    else if constexpr (triangular_adapter<A> and triangle_type_of_v<A> == triangle_type_of_v<A, B>)
    {
      internal::set_triangle<triangle_type_of_v<A>>(a, contract(a, std::forward<B>(b)));
      return a;
    }
#ifdef __cpp_concepts
    else if constexpr (requires { interface::LinearAlgebra<std::decay_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b)); })
#else
    else if constexpr (detail::contract_in_place_exists<on_the_right, A, B>::value)
#endif
    {
      return interface::LinearAlgebra<std::decay_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b));
    }
    else
    {
      a = contract(a, std::forward<B>(b));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_LINEAR_ALGEBRA_FUNCTIONS_HPP
