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
 * \brief Overloaded general functions relating to scalar types.
 */

#ifndef OPENKALMAN_SCALAR_FUNCTIONS_HPP
#define OPENKALMAN_SCALAR_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Project to a real number of a \ref std::floating_point type that depends on the argument.
   * \tparam Arg a \ref scalar_type
   * \details For example, if the argument is a complex number, the function will convert to the type of its real part.
   */
#ifdef __cpp_concepts
  constexpr std::floating_point auto real_projection(scalar_type auto&& arg)
#else
  template<typename Arg, std::enable_if_t<scalar_type<Arg>, int> = 0>
  constexpr auto real_projection(Arg&& arg)
#endif
  {
    return interface::ScalarTraits<std::decay_t<decltype(arg)>>::template real_projection(std::forward<decltype(arg)>(arg));
  }


  /**
   * \overload
   * \brief Project to a real number of type Scalar.
   * \details This is used for wrapping the real part of angles or other modular scalar values. If the argument is
   * already a real number (as opposed, for example, to a complex number), this will be an identity function
   * (if already \ref std::floating_point) or a numerical conversion from a \ref std::integral or custom scalar type to
   * \ref std::floating_point.
   * \tparam Scalar a std::floating_point type to convert to.
   * \tparam Arg a \ref scalar_type
   * \return A number of type Scalar. This will be the real part of a complex number, or the argument converted to Scalar.
   */
#ifdef __cpp_concepts
  template<std::floating_point Scalar>
  constexpr Scalar real_projection(scalar_type auto&& arg)
  requires std::convertible_to<std::decay_t<decltype(real_projection(arg))>, Scalar>
#else
  template<typename Scalar, typename Arg, std::enable_if_t<std::is_floating_point_v<Scalar> and scalar_type<Arg> and
    std::is_convertible_v<decltype(real_projection(std::declval<Arg&&>())), std::decay_t<Scalar>>, int> = 0>
  constexpr Scalar real_projection(Arg&& arg)
#endif
  {
    return real_projection(std::forward<decltype(arg)>(arg));
  }


  /**
   * \brief The inverse of \ref real_projection.
   * \details This takes a real number (\ref std::floating_point) and recovers a corresponding scalar value
   * from which it would have been a projection. This function must obey the following identity for all
   * <code>x</code> of type Scalar: <code>x == inverse_real_projection(x, real_projection(x))</code>.
   * For example, if the argument is a complex number, the result of this function is a complex number whose real
   * part is updated with the value p of floating type RealProj.
   * \tparam Scalar a \ref scalar_type
   * \param real_projection A \ref std::floating_point argument representing a hypothetical result of \ref real_projection.
   */
#ifdef __cpp_concepts
  constexpr floating_scalar_type auto
  inverse_real_projection(floating_scalar_type auto&& arg, std::decay_t<decltype(real_projection(arg))> real_projection)
#else
  template<typename Scalar, std::enable_if_t<floating_scalar_type<Scalar>, int> = 0>
  constexpr auto inverse_real_projection(Scalar&& arg, std::decay_t<decltype(real_projection(arg))> real_projection)
#endif
  {
    return interface::ScalarTraits<std::decay_t<decltype(arg)>>::template inverse_real_projection(
      std::forward<decltype(arg)>(arg), real_projection);
  }

} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_FUNCTIONS_HPP
